"""Centre de simulation : lance les missions contre les simulateurs, un essai ou une campagne.

    python3 test/sim_hub.py                              interface graphique
    python3 test/sim_hub.py --batch [--seeds 5] [--level real,hard] [--only enclos,piste]
                                    [--jobs 6]           campagne en terminal (code retour 0 = tout OK)
    python3 test/sim_hub.py --run piste-aleatoire --seed 3 --level hard [--view]   un seul essai
    python3 test/sim_hub.py --trajets [DOSSIER]          tous les trajets d'une campagne sur un écran

Depuis le PC (hors Docker) : test/sim_hub.sh [mêmes options].

Chaque essai tourne avec son propre ROS_DOMAIN_ID (60-99) et ROS_LOCALHOST_ONLY=1 : on peut
lancer plusieurs essais en parallèle, et rien ne part vers le vrai robot (domaine 39).
Les résultats et les logs vont dans test/results/<date>/ (résumé dans results.csv).
"""
import argparse
import csv
import itertools
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
WS = PKG.parent.parent
CONFIG = PKG / 'config' / 'robothub.yaml'
TEMPLATES = Path('/tmp/robothub_sim_templates')
RESULTS = HERE / 'results'
LEVELS = ['ideal', 'real', 'hard']
TIMEOUT = 200  # s réels par essai, au-delà : tué


@dataclass
class Scenario:
    name: str
    label: str
    mission: str      # commande de la mission (ros2 run ...)
    sim: str          # commande du simulateur, {seed} {level} remplacés
    group: str


def _track(mission):
    return f'ros2 run robothub track_mission --ros-args --params-file {CONFIG} -p mission:={mission}'


SCENARIOS = [
    Scenario('piste-droite', 'Piste : ligne droite + damier', _track('straight'),
             'fake_track_sim.py straight', 'piste'),
    Scenario('piste-virage', 'Piste : virage + panneau STOP', _track('curve'),
             'fake_track_sim.py curve', 'piste'),
    Scenario('piste-aleatoire', 'Piste : tracé aléatoire (S) + damier', _track('straight'),
             'fake_track_sim.py random', 'piste'),
    Scenario('barriere', 'Barrière : arrêt puis reprise', _track('barrier'),
             'fake_track_sim.py straight --barrier', 'piste'),
    Scenario('intersection', 'Intersection : priorité à droite', _track('intersection'),
             'fake_track_sim.py straight --robot-right', 'piste'),
    Scenario('carrefour', 'Carrefour fig. 3 : tout droit', _track('intersection'),
             'fake_track_sim.py cross', 'piste'),
    Scenario('carrefour-priorite', 'Carrefour fig. 3 + robot 2 à droite', _track('intersection'),
             'fake_track_sim.py cross --robot-right', 'piste'),
    Scenario('parcours', 'Parcours : barrière + carrefour tout droit + virage + STOP', _track('parcours'),
             'fake_track_sim.py parcours', 'piste'),
    Scenario('parcours-priorite', 'Parcours + robot 2 à droite au carrefour', _track('parcours'),
             'fake_track_sim.py parcours --robot-right', 'piste'),
    Scenario('enclos-fixe', 'Enclos : figure 4 du règlement',
             f'ros2 run robothub enclosure_mission --ros-args --params-file {CONFIG}',
             'fake_enclosure_sim.py --layout fixed', 'enclos'),
    Scenario('enclos-aleatoire', 'Enclos : forme et portes aléatoires',
             f'ros2 run robothub enclosure_mission --ros-args --params-file {CONFIG}',
             'fake_enclosure_sim.py --layout random', 'enclos'),
    Scenario('enclos-puis-piste', 'Enclos (LiDAR) puis suivi de ligne jusqu\'au damier',
             f'ros2 run robothub enclosure_mission --ros-args --params-file {CONFIG} -p handoff:=true -p exit_extra:=0.05 & '
             f'ros2 run robothub track_mission --ros-args --params-file {CONFIG} -p mission:=straight '
             f'-p start_topic:=/robothub/enclosure_done -p start_delay:=0.3 & wait',
             'fake_enclosure_sim.py --layout fixed --track', 'enclos'),
    Scenario('stockage', 'Stockage : palette ① -> ②',
             f'ros2 run robothub storage_mission --ros-args --params-file {CONFIG} '
             f'-p templates_dir:={TEMPLATES} -p sign_dist:=0.55',
             'fake_storage_sim.py', 'stockage'),
]
BY_NAME = {s.name: s for s in SCENARIOS}


@dataclass
class Result:
    scenario: str
    seed: int
    level: str
    status: str = 'EN COURS'   # SUCCESS / ECHEC / COLLISION / TIMEOUT / ... / ERREUR
    detail: str = ''
    duration: float = 0.0
    log_dir: str = ''
    events: list = field(default_factory=list)

    @property
    def ok(self):
        return self.status == 'SUCCESS'


def ros_shell(cmd):
    """Commande lancée avec l'environnement ROS + le workspace sourcés."""
    setup = WS / 'install' / 'setup.bash'
    src = 'source /opt/ros/humble/setup.bash' + (f' && source {setup}' if setup.exists() else '')
    return ['bash', '-c', f'{src} && ' + (cmd if ' & ' in cmd else f'exec {cmd}')]


def ensure_templates():
    if not TEMPLATES.exists() or not list(TEMPLATES.glob('*.png')):
        subprocess.run([sys.executable, str(HERE / 'fake_storage_sim.py'), '--make-templates', str(TEMPLATES)],
                       check=True, stdout=subprocess.DEVNULL)


_domains = queue.Queue()
for _d in range(60, 100):
    _domains.put(_d)


def run_one(scn, seed, level, view=False, out_dir=None, on_line=None, stop_event=None):
    """Lance mission + simulateur, attend le verdict du simulateur. Bloquant."""
    if scn.group == 'stockage':
        ensure_templates()
    res = Result(scn.name, seed, level)
    out_dir = Path(out_dir or RESULTS / 'unitaire')
    log_dir = out_dir / f'{scn.name}_s{seed}_{level}'
    log_dir.mkdir(parents=True, exist_ok=True)
    res.log_dir = str(log_dir)
    domain = _domains.get()
    env = dict(os.environ, ROS_DOMAIN_ID=str(domain), ROS_LOCALHOST_ONLY='1', PYTHONUNBUFFERED='1')
    if not view:
        env.pop('SIM_VIEW_DUMP', None)
    env['SIM_TRAJ'] = str(log_dir / 'traj.jsonl')   # trajectoire vraie, pour la vue d'ensemble (sim_trajets.py)
    sim_cmd = f'python3 {HERE / scn.sim} --seed {seed} --level {level}' + (' --view' if view else '')
    t0 = time.time()
    mission = sim = None
    try:
        with open(log_dir / 'mission.log', 'w') as mlog:
            mission = subprocess.Popen(ros_shell(scn.mission), stdout=mlog, stderr=subprocess.STDOUT,
                                       env=env, start_new_session=True, cwd=PKG)
            sim = subprocess.Popen(ros_shell(sim_cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   env=env, start_new_session=True, cwd=HERE, text=True, bufsize=1)
            last = ''
            with open(log_dir / 'sim.log', 'w') as slog:
                killer = threading.Timer(TIMEOUT, lambda: _kill(sim))
                killer.start()
                stopper = None
                if stop_event is not None:
                    stopper = threading.Thread(target=lambda: (stop_event.wait(), _kill(sim)), daemon=True)
                    stopper.start()
                for line in sim.stdout:
                    slog.write(line)
                    line = line.rstrip()
                    if line:
                        last = line
                        if on_line:
                            on_line(line)
                        if re.match(r't=\s*[\d.]+ ', line):
                            res.events.append(line)
                sim.wait()
                killer.cancel()
        m = re.match(r'(SUCCESS|ECHEC|COLLISION|TIMEOUT|SORTIE DE PISTE|BARRIÈRE PERCUTÉE|MAUVAISE DIRECTION)\b\s*:?\s*(.*)', last)
        if m:
            res.status, res.detail = m.group(1), m.group(2)
        elif stop_event is not None and stop_event.is_set():
            res.status, res.detail = 'ARRÊTÉ', ''
        else:
            res.status, res.detail = 'ERREUR', last[-160:]
            tail = (log_dir / 'mission.log').read_text(errors='replace').strip().splitlines()[-3:]
            if tail:
                res.detail += ' | mission : ' + ' / '.join(tail)[-200:]
    finally:
        for p in (sim, mission):
            if p is not None:
                _kill(p)
        _domains.put(domain)
    res.duration = time.time() - t0
    return res


def _kill(p):
    """Arrête tout le groupe de processus de `p`. Ctrl+C d'abord (la mission publie vitesse nulle),
    puis on s'assure qu'il ne reste personne : dans une commande « A & B & wait », A et B ignorent
    Ctrl+C et survivaient à bash — une mission fantôme restait sur son ROS_DOMAIN_ID et envoyait
    des ordres d'arrêt au robot de l'essai suivant qui réutilisait ce domaine."""
    def alive():
        try:
            os.killpg(p.pid, 0)
            return True
        except (ProcessLookupError, PermissionError):
            return False
    if p.poll() is None:
        try:
            os.killpg(p.pid, signal.SIGINT)
            p.wait(timeout=3)
        except Exception:
            pass
    for sig, wait in ((signal.SIGTERM, 1.0), (signal.SIGKILL, 0.5)):
        if not alive():
            break
        try:
            os.killpg(p.pid, sig)
        except ProcessLookupError:
            break
        t_end = time.time() + wait
        while alive() and time.time() < t_end:
            time.sleep(0.05)
    try:
        p.wait(timeout=1)
    except Exception:
        pass


def plan(scenarios, seeds, levels):
    return [(BY_NAME[s], seed, lv) for s, lv, seed in itertools.product(scenarios, levels, seeds)]


def write_csv(path, results):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['scenario', 'seed', 'niveau', 'resultat', 'detail', 'duree_s', 'logs'])
        for r in results:
            w.writerow([r.scenario, r.seed, r.level, r.status, r.detail, f'{r.duration:.1f}', r.log_dir])


def summary(results):
    lines = []
    rows = sorted({(r.scenario, r.level) for r in results}, key=lambda k: (list(BY_NAME).index(k[0]), k[1]))
    lines.append(f'{"scénario":<18} {"niveau":<6} {"réussite":>9}  échecs (seed : cause)')
    for scn, lv in rows:
        rs = [r for r in results if r.scenario == scn and r.level == lv]
        ok = sum(r.ok for r in rs)
        fails = ', '.join(f'{r.seed}:{r.status}' for r in rs if not r.ok)
        lines.append(f'{scn:<18} {lv:<6} {ok:>4}/{len(rs):<4}  {fails}')
    total = sum(r.ok for r in results)
    lines.append(f'TOTAL {total}/{len(results)}')
    return '\n'.join(lines)


# --------------------------------------------------------------------------- terminal

def batch_cli(args):
    scen = [s.name for s in SCENARIOS
            if not args.only or any(o in s.name or o == s.group for o in args.only.split(','))]
    levels = args.level.split(',')
    jobs = plan(scen, range(args.seed0, args.seed0 + args.seeds), levels)
    out = RESULTS / time.strftime('%Y%m%d-%H%M%S')
    print(f'{len(jobs)} essais ({len(scen)} scénarios x {args.seeds} seeds x {levels}), '
          f'{args.jobs} en parallèle -> {out}', flush=True)
    results = []
    with ThreadPoolExecutor(args.jobs) as ex:
        futs = [ex.submit(run_one, s, seed, lv, False, out) for s, seed, lv in jobs]
        for f in futs:
            r = f.result()
            results.append(r)
            mark = 'OK ' if r.ok else '!! '
            print(f'{mark}{len(results):>3}/{len(jobs)} {r.scenario:<18} seed={r.seed:<3} {r.level:<5} '
                  f'{r.status} {r.detail[:90]}', flush=True)
    write_csv(out / 'results.csv', results)
    print('\n' + summary(results))
    print(f'\nlogs : {out}')
    return 0 if all(r.ok for r in results) else 1


# --------------------------------------------------------------------------- interface graphique

def gui():
    import tkinter as tk
    from tkinter import messagebox, ttk

    root = tk.Tk()
    root.title('RobotHub — simulations')
    root.geometry('1180x720')
    style = ttk.Style(root)
    if 'clam' in style.theme_names():
        style.theme_use('clam')
    style.configure('Treeview', rowheight=22)
    style.configure('Big.TButton', font=('TkDefaultFont', 11, 'bold'), padding=6)
    events = queue.Queue()
    state = {'stop': None, 'results': {}, 'running': 0}

    paned = ttk.PanedWindow(root, orient='horizontal')
    paned.pack(fill='both', expand=True, padx=8, pady=8)

    # ---------------- un essai
    left = ttk.LabelFrame(paned, text=' Un essai (avec la vue) ', padding=8)
    paned.add(left, weight=1)
    ttk.Label(left, text='Scénario').grid(row=0, column=0, sticky='w')
    scn_var = tk.StringVar(value=SCENARIOS[0].label)
    ttk.Combobox(left, textvariable=scn_var, values=[s.label for s in SCENARIOS], state='readonly',
                 width=38).grid(row=0, column=1, columnspan=3, sticky='we', pady=2)
    ttk.Label(left, text='Seed').grid(row=1, column=0, sticky='w')
    seed_var = tk.IntVar(value=0)
    ttk.Spinbox(left, from_=0, to=9999, textvariable=seed_var, width=8).grid(row=1, column=1, sticky='w', pady=2)
    ttk.Label(left, text='Réalisme').grid(row=2, column=0, sticky='w')
    lvl_var = tk.StringVar(value='real')
    for i, lv in enumerate(LEVELS):
        ttk.Radiobutton(left, text=lv, value=lv, variable=lvl_var).grid(row=2, column=1 + i, sticky='w')
    view_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(left, text='Afficher la vue de dessus', variable=view_var).grid(row=3, column=1, columnspan=3,
                                                                                    sticky='w', pady=2)
    run_btn = ttk.Button(left, text='▶  Lancer', style='Big.TButton')
    run_btn.grid(row=4, column=0, columnspan=4, sticky='we', pady=6)
    verdict = ttk.Label(left, text='', font=('TkDefaultFont', 12, 'bold'))
    verdict.grid(row=5, column=0, columnspan=4, sticky='w')
    log = tk.Text(left, height=20, width=60, font=('TkFixedFont', 9), bg='#1e1e1e', fg='#d0d0d0',
                  insertbackground='white')
    log.grid(row=6, column=0, columnspan=4, sticky='nsew')
    left.rowconfigure(6, weight=1)
    left.columnconfigure(3, weight=1)
    help_txt = ('idéal : robot parfait (ancien simulateur)\n'
                'real : accélérations, retards, odométrie qui dérive, LiDAR et caméra bruités\n'
                'hard : pire cas (Wi-Fi chargé, éclairage changeant, gros décalage au départ)')
    ttk.Label(left, text=help_txt, foreground='#666').grid(row=7, column=0, columnspan=4, sticky='w', pady=4)

    def label_to_scn(label):
        return next(s for s in SCENARIOS if s.label == label)

    def run_single(scn=None, seed=None, level=None):
        scn = scn or label_to_scn(scn_var.get())
        seed = seed_var.get() if seed is None else seed
        level = level or lvl_var.get()
        log.delete('1.0', 'end')
        verdict.config(text=f'en cours : {scn.name} seed={seed} {level}…', foreground='#555')
        run_btn.state(['disabled'])

        def work():
            r = run_one(scn, seed, level, view=view_var.get(),
                        on_line=lambda line: events.put(('log', line)))
            events.put(('single_done', r))
        threading.Thread(target=work, daemon=True).start()

    run_btn.config(command=run_single)

    # ---------------- campagne
    right = ttk.LabelFrame(paned, text=' Campagne (sans vue, en parallèle) ', padding=8)
    paned.add(right, weight=2)
    opts = ttk.Frame(right)
    opts.pack(fill='x')
    scn_vars = {}
    box = ttk.Frame(opts)
    box.grid(row=0, column=0, rowspan=4, sticky='nw', padx=(0, 16))
    for i, s in enumerate(SCENARIOS):
        v = tk.BooleanVar(value=True)
        scn_vars[s.name] = v
        ttk.Checkbutton(box, text=s.name, variable=v).grid(row=i % 4, column=i // 4, sticky='w', padx=(0, 10))
    ttk.Label(opts, text='Seeds').grid(row=0, column=1, sticky='w')
    nseeds = tk.IntVar(value=3)
    ttk.Spinbox(opts, from_=1, to=100, textvariable=nseeds, width=5).grid(row=0, column=2, sticky='w')
    ttk.Label(opts, text='Parallèle').grid(row=1, column=1, sticky='w')
    njobs = tk.IntVar(value=max(2, min(8, (os.cpu_count() or 4) // 2)))
    ttk.Spinbox(opts, from_=1, to=32, textvariable=njobs, width=5).grid(row=1, column=2, sticky='w')
    lvl_vars = {}
    lv_box = ttk.Frame(opts)
    lv_box.grid(row=2, column=1, columnspan=2, sticky='w')
    for lv in LEVELS:
        lvl_vars[lv] = tk.BooleanVar(value=lv != 'ideal')
        ttk.Checkbutton(lv_box, text=lv, variable=lvl_vars[lv]).pack(side='left')
    btns = ttk.Frame(opts)
    btns.grid(row=3, column=1, columnspan=2, sticky='w', pady=4)
    batch_btn = ttk.Button(btns, text='▶  Lancer la campagne', style='Big.TButton')
    batch_btn.pack(side='left')
    stop_btn = ttk.Button(btns, text='■ Arrêter', state='disabled')
    stop_btn.pack(side='left', padx=6)

    def show_trajets():
        import sim_trajets
        w = state.get('trajets')
        if w is not None and w.winfo_exists():
            w.lift()
            return
        state['trajets'] = sim_trajets.open_window(root, state.get('out'), follow=lambda: state.get('out'))
    ttk.Button(btns, text='🗺  Voir les trajets', command=show_trajets).pack(side='left', padx=6)

    progress = ttk.Progressbar(right, mode='determinate')
    progress.pack(fill='x', pady=(8, 2))
    prog_lbl = ttk.Label(right, text='')
    prog_lbl.pack(anchor='w')

    cols = ('scenario', 'seed', 'niveau', 'resultat', 'duree', 'detail')
    tree = ttk.Treeview(right, columns=cols, show='headings')
    for c, w, t in zip(cols, (130, 45, 55, 120, 55, 420),
                       ('Scénario', 'Seed', 'Niveau', 'Résultat', 'Durée', 'Détail')):
        tree.heading(c, text=t)
        tree.column(c, width=w, anchor='w', stretch=c == 'detail')
    tree.tag_configure('ok', background='#d9f2d9')
    tree.tag_configure('fail', background='#f8d4d4')
    tree.tag_configure('run', background='#fff4cc')
    sb = ttk.Scrollbar(right, orient='vertical', command=tree.yview)
    tree.configure(yscrollcommand=sb.set)
    tree.pack(side='left', fill='both', expand=True, pady=4)
    sb.pack(side='left', fill='y', pady=4)
    ttk.Label(root, text='Double-clic sur une ligne : rejouer cet essai avec la vue.   '
                         'Clic droit : ouvrir les logs dans le panneau de gauche.',
              foreground='#666').pack(anchor='w', padx=10, pady=(0, 6))
    summary_lbl = ttk.Label(root, text='', font=('TkFixedFont', 9), justify='left')

    def start_batch():
        scen = [n for n, v in scn_vars.items() if v.get()]
        levels = [lv for lv, v in lvl_vars.items() if v.get()]
        if not scen or not levels:
            messagebox.showwarning('Campagne', 'Choisis au moins un scénario et un niveau.')
            return
        jobs = plan(scen, range(nseeds.get()), levels)
        tree.delete(*tree.get_children())
        state['results'] = {}
        out = RESULTS / time.strftime('%Y%m%d-%H%M%S')
        state['out'] = out
        stop = threading.Event()
        state['stop'] = stop
        progress.config(maximum=len(jobs), value=0)
        prog_lbl.config(text=f'0/{len(jobs)}')
        batch_btn.state(['disabled'])
        stop_btn.state(['!disabled'])
        for s, seed, lv in jobs:
            iid = f'{s.name}|{seed}|{lv}'
            tree.insert('', 'end', iid=iid, values=(s.name, seed, lv, 'en attente', '', ''))

        def work():
            with ThreadPoolExecutor(njobs.get()) as ex:
                def one(s, seed, lv):
                    if stop.is_set():
                        return Result(s.name, seed, lv, 'ANNULÉ')
                    events.put(('started', f'{s.name}|{seed}|{lv}'))
                    return run_one(s, seed, lv, False, out, stop_event=stop)
                futs = [ex.submit(one, s, seed, lv) for s, seed, lv in jobs]
                for f in futs:
                    events.put(('batch_result', f.result()))
            events.put(('batch_done', len(jobs)))
        threading.Thread(target=work, daemon=True).start()

    batch_btn.config(command=start_batch)
    stop_btn.config(command=lambda: state['stop'] and state['stop'].set())

    def on_double(_e):
        sel = tree.focus()
        if not sel:
            return
        name, seed, lv = sel.split('|')
        view_var.set(True)
        scn_var.set(BY_NAME[name].label)
        seed_var.set(int(seed))
        lvl_var.set(lv)
        run_single(BY_NAME[name], int(seed), lv)

    def on_right(e):
        iid = tree.identify_row(e.y)
        r = state['results'].get(iid)
        if not r or not r.log_dir:
            return
        log.delete('1.0', 'end')
        for f in ('sim.log', 'mission.log'):
            p = Path(r.log_dir) / f
            if p.exists():
                log.insert('end', f'===== {p}\n', 'h')
                log.insert('end', p.read_text(errors='replace')[-6000:] + '\n')
        log.tag_config('h', foreground='#7fc7ff')
        verdict.config(text=f'logs : {iid}  ->  {r.status}', foreground='#333')

    tree.bind('<Double-1>', on_double)
    tree.bind('<Button-3>', on_right)

    def poll():
        try:
            handle_events()
        except Exception as e:  # une erreur ne doit pas figer l'interface
            import traceback
            traceback.print_exc()
            verdict.config(text=f'erreur interne : {e}', foreground='#b00000')
        root.after(100, poll)

    def handle_events():
        try:
            while True:
                kind, data = events.get_nowait()
                if kind == 'log':
                    tag = 'ok' if data.startswith('SUCCESS') else 'bad' if re.match(
                        r'(ECHEC|COLLISION|TIMEOUT|SORTIE|BARRIÈRE PERCUTÉE)', data) else None
                    log.insert('end', data + '\n', tag or ())
                    log.tag_config('ok', foreground='#6fdc6f')
                    log.tag_config('bad', foreground='#ff7070')
                    log.see('end')
                elif kind == 'single_done':
                    r = data
                    run_btn.state(['!disabled'])
                    verdict.config(text=f'{r.status}  {r.detail[:70]}',
                                   foreground='#1a7f1a' if r.ok else '#b00000')
                elif kind == 'started':
                    if tree.exists(data):
                        tree.set(data, 'resultat', 'en cours…')
                        tree.item(data, tags=('run',))
                elif kind == 'batch_result':
                    r = data
                    iid = f'{r.scenario}|{r.seed}|{r.level}'
                    state['results'][iid] = r
                    if tree.exists(iid):
                        tree.item(iid, values=(r.scenario, r.seed, r.level, r.status, f'{r.duration:.0f}s',
                                               r.detail), tags=('ok' if r.ok else 'fail',))
                    done = len(state['results'])
                    ok = sum(x.ok for x in state['results'].values())
                    progress.config(value=done)
                    prog_lbl.config(text=f'{done}/{int(progress["maximum"])}  —  {ok} réussis, {done - ok} échecs')
                elif kind == 'batch_done':
                    batch_btn.state(['!disabled'])
                    stop_btn.state(['disabled'])
                    res = list(state['results'].values())
                    write_csv(state['out'] / 'results.csv', res)
                    summary_lbl.config(text=summary(res))
                    summary_lbl.pack(anchor='w', padx=10, pady=(0, 6))
                    prog_lbl.config(text=prog_lbl.cget('text') + f'   (résultats : {state["out"]})')
        except queue.Empty:
            pass

    root.hub = dict(tree=tree, summary=summary_lbl, progress=prog_lbl, start_batch=start_batch,
                    run_single=run_single, verdict=verdict, log=log)  # pour les tests
    poll()
    root.mainloop()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--batch', action='store_true')
    ap.add_argument('--run', choices=list(BY_NAME))
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--seed0', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--level', default='real,hard')
    ap.add_argument('--only', default='')
    ap.add_argument('--jobs', type=int, default=max(2, min(8, (os.cpu_count() or 4) // 2)))
    ap.add_argument('--view', action='store_true')
    ap.add_argument('--trajets', nargs='?', const='', metavar='DOSSIER',
                    help='vue d\'ensemble des trajets d\'une campagne (défaut : la dernière)')
    a = ap.parse_args()
    if a.trajets is not None:
        import sim_trajets
        return sim_trajets.main(a.trajets or None)
    if a.batch:
        sys.exit(batch_cli(a))
    if a.run:
        r = run_one(BY_NAME[a.run], a.seed, a.level.split(',')[0], a.view, on_line=print)
        print(f'\n=> {r.status} {r.detail}\nlogs : {r.log_dir}')
        sys.exit(0 if r.ok else 1)
    gui()


if __name__ == '__main__':
    main()
