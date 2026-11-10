"""Vue d'ensemble des trajets : toutes les simulations d'une campagne en même temps.

    test/sim_hub.sh --trajets              dernière campagne (suit aussi une campagne en cours)
    test/sim_hub.sh --trajets DOSSIER      une campagne précise (test/results/<date>)
    bouton « Voir les trajets » du centre de simulation

Une vignette par terrain : pour les scénarios à terrain fixe, tous les essais sont superposés ;
pour les terrains aléatoires, une vignette par seed (niveaux real et hard superposés).
Vert = réussi, rouge = échec (croix à l'endroit de la fin), bleu = en cours, gris = interrompu.
Survoler un trajet : seed, niveau et verdict. Cliquer : le garder en surbrillance.

Les simulateurs écrivent la vraie position du robot dans <essai>/traj.jsonl (voir sim_core.TrajLog).
"""
import json
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / 'results'
COLORS = {'ok': '#1f9d3a', 'fail': '#d62828', 'run': '#1d6fd8', 'dead': '#8a8a8a'}
PANEL_W, PANEL_H = 400, 300
DIR_RE = re.compile(r'(.+)_s(\d+)_([a-z]+)$')


class Run:
    """Un essai : lit son traj.jsonl au fur et à mesure qu'il grossit."""

    def __init__(self, path):
        self.path = path
        m = DIR_RE.match(path.parent.name)
        self.scenario, self.seed, self.level = (m.group(1), int(m.group(2)), m.group(3)) if m else ('?', 0, '?')
        self.pos = 0
        self.meta = None
        self.pts = []
        self.end = None
        self.mtime = 0.0

    def refresh(self):
        """-> True si quelque chose de nouveau a été lu."""
        try:
            st = self.path.stat()
        except OSError:
            return False
        self.mtime = st.st_mtime
        if st.st_size <= self.pos:
            return False
        with open(self.path, 'rb') as f:
            f.seek(self.pos)
            data = f.read()
        last_nl = data.rfind(b'\n')
        if last_nl < 0:
            return False
        self.pos += last_nl + 1
        for line in data[:last_nl].decode('utf-8', 'replace').splitlines():
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if isinstance(d, list):
                self.pts.append((d[1], d[2]))
            elif 'meta' in d:
                self.meta = d['meta']
            elif 'end' in d:
                self.end = d['end']
        return True

    def status(self, campaign_done):
        if self.end is not None:
            return 'ok' if self.end.startswith('SUCCESS') else 'fail'
        return 'dead' if campaign_done or time.time() - self.mtime > 15 else 'run'

    @property
    def key(self):
        """Essais superposables = même scénario et même terrain."""
        shapes = json.dumps(self.meta.get('shapes', []), sort_keys=True) if self.meta else ''
        return self.scenario, hash(shapes)

    @property
    def label(self):
        verdict = self.end if self.end is not None else 'en cours / interrompu'
        return f'{self.scenario}  seed {self.seed}  {self.level}  —  {verdict}'


def campaigns():
    if not RESULTS.exists():
        return []
    return sorted((d for d in RESULTS.iterdir() if d.is_dir()), key=lambda d: d.stat().st_mtime, reverse=True)


class Viewer:
    def __init__(self, parent, directory=None, follow=None):
        """`follow` : fonction renvoyant le dossier de la campagne en cours du centre de simulation
        (ou None) ; la vue bascule dessus dès qu'une campagne démarre."""
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.ttk = tk, ttk
        self.win = parent
        self.follow = follow
        self.runs = {}
        self.panels = {}       # key -> dict(canvas, frame, title, dirty)
        self.picked = None
        self.dir = None
        self.cols = 0

        top = ttk.Frame(parent, padding=(8, 6))
        top.pack(fill='x')
        ttk.Label(top, text='Campagne').pack(side='left')
        self.dir_var = tk.StringVar()
        self.dir_box = ttk.Combobox(top, textvariable=self.dir_var, state='readonly', width=22,
                                    postcommand=self.fill_dirs)
        self.dir_box.pack(side='left', padx=(4, 12))
        self.dir_box.bind('<<ComboboxSelected>>', lambda e: self.set_dir(RESULTS / self.dir_var.get()))
        ttk.Label(top, text='Scénario').pack(side='left')
        self.scn_var = tk.StringVar(value='tous')
        self.scn_box = ttk.Combobox(top, textvariable=self.scn_var, state='readonly', width=20, values=['tous'])
        self.scn_box.pack(side='left', padx=(4, 12))
        self.scn_box.bind('<<ComboboxSelected>>', lambda e: self.rebuild())
        self.show = {}
        for k, text in (('ok', 'réussis'), ('fail', 'échecs'), ('run', 'en cours'), ('real', 'real'),
                        ('hard', 'hard'), ('ideal', 'ideal')):
            self.show[k] = tk.BooleanVar(value=True)
            cb = tk.Checkbutton(top, text=text, variable=self.show[k], command=self.redraw_all,
                                fg=COLORS.get(k, '#222'), activeforeground=COLORS.get(k, '#222'))
            cb.pack(side='left')
        self.only_fail = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text='terrains avec échec seulement', variable=self.only_fail,
                        command=self.rebuild).pack(side='left', padx=(12, 0))
        self.count_lbl = ttk.Label(top, text='')
        self.count_lbl.pack(side='right')

        self.info = ttk.Label(parent, text='Survole un trajet pour voir de quel essai il s\'agit.',
                              padding=(8, 2), font=('TkDefaultFont', 10, 'bold'))
        self.info.pack(fill='x')

        body = ttk.Frame(parent)
        body.pack(fill='both', expand=True)
        self.outer = tk.Canvas(body, highlightthickness=0, bg='#ececec')
        sb = ttk.Scrollbar(body, orient='vertical', command=self.outer.yview)
        self.outer.configure(yscrollcommand=sb.set)
        sb.pack(side='right', fill='y')
        self.outer.pack(side='left', fill='both', expand=True)
        self.grid = tk.Frame(self.outer, bg='#ececec')
        self.outer.create_window(0, 0, window=self.grid, anchor='nw')
        self.grid.bind('<Configure>', lambda e: self.outer.configure(scrollregion=self.outer.bbox('all')))
        self.outer.bind('<Configure>', self.on_resize)
        for seq, d in (('<Button-4>', -3), ('<Button-5>', 3)):
            parent.bind_all(seq, lambda e, d=d: self.outer.yview_scroll(d, 'units'))
        parent.bind_all('<MouseWheel>', lambda e: self.outer.yview_scroll(-e.delta // 40, 'units'))

        cs = campaigns()
        self.set_dir(Path(directory) if directory else (cs[0] if cs else None))
        self.tick()

    # ------------------------------------------------------------------ données
    def fill_dirs(self):
        self.dir_box['values'] = [d.name for d in campaigns()][:60]

    def set_dir(self, d):
        self.dir = d
        self.dir_var.set(d.name if d else '')
        self.runs = {}
        self.picked = None
        self.rebuild()

    def scan(self):
        """Nouveaux essais + nouveaux points. -> clés des vignettes à redessiner, et s'il faut tout reconstruire."""
        if self.dir is None or not self.dir.exists():
            return set(), False
        changed, new_group = set(), False
        for sub in os.scandir(self.dir):
            if not sub.is_dir():
                continue
            p = Path(sub.path) / 'traj.jsonl'
            run = self.runs.get(sub.name)
            if run is None:
                if not p.exists():
                    continue
                run = self.runs[sub.name] = Run(p)
            had_meta = run.meta is not None
            if run.refresh():
                if run.meta is None:
                    continue
                if not had_meta and run.key not in self.panels:
                    new_group = True
                changed.add(run.key)
        return changed, new_group

    def campaign_done(self):
        return self.dir is not None and (self.dir / 'results.csv').exists()

    # ------------------------------------------------------------------ vignettes
    def groups(self):
        out = {}
        for r in self.runs.values():
            if r.meta is not None:
                out.setdefault(r.key, []).append(r)
        return out

    def rebuild(self):
        for p in self.panels.values():
            p['frame'].destroy()
        self.panels = {}
        groups = self.groups()
        names = sorted({k[0] for k in groups})
        self.scn_box['values'] = ['tous'] + names
        if self.scn_var.get() not in self.scn_box['values']:
            self.scn_var.set('tous')
        per_scn = {}
        for k in groups:
            per_scn[k[0]] = per_scn.get(k[0], 0) + 1
        done = self.campaign_done()
        keys = sorted(groups, key=lambda k: (k[0], min(r.seed for r in groups[k])))
        for key in keys:
            if self.scn_var.get() not in ('tous', key[0]):
                continue
            if self.only_fail.get() and not any(r.status(done) == 'fail' for r in groups[key]):
                continue
            frame = self.tk.Frame(self.grid, bg='white', highlightbackground='#bbb', highlightthickness=1)
            title = self.tk.Label(frame, text='', bg='white', anchor='w', font=('TkDefaultFont', 9, 'bold'))
            title.pack(fill='x', padx=6, pady=(4, 0))
            canvas = self.tk.Canvas(frame, width=PANEL_W, height=PANEL_H, bg='#f7f7f4', highlightthickness=0)
            canvas.pack(padx=4, pady=4)
            self.panels[key] = dict(frame=frame, title=title, canvas=canvas, many=per_scn[key[0]] > 1)
        self.place()
        self.redraw_all()

    def place(self):
        width = max(self.outer.winfo_width(), PANEL_W + 20)
        cols = max(1, width // (PANEL_W + 18))
        self.cols = cols
        for i, p in enumerate(self.panels.values()):
            p['frame'].grid(row=i // cols, column=i % cols, padx=5, pady=5)

    def on_resize(self, _e):
        if max(1, self.outer.winfo_width() // (PANEL_W + 18)) != self.cols:
            self.place()

    def redraw_all(self):
        groups = self.groups()
        for key in self.panels:
            self.draw(key, groups.get(key, []))
        self.update_count()

    def update_count(self):
        done = self.campaign_done()
        st = [r.status(done) for r in self.runs.values() if r.meta is not None]
        self.count_lbl.config(text=f'{len(st)} essais : {st.count("ok")} réussis, {st.count("fail")} échecs, '
                                   f'{st.count("run")} en cours')

    def draw(self, key, runs):
        p = self.panels[key]
        c = p['canvas']
        c.delete('all')
        if not runs:
            return
        done = self.campaign_done()
        meta = runs[0].meta
        x0, y0, x1, y1 = meta['bounds']
        for r in runs:   # un robot sorti du cadre prévu reste visible
            if r.pts:
                xs, ys = [q[0] for q in r.pts], [q[1] for q in r.pts]
                x0, y0, x1, y1 = min(x0, min(xs) - 0.05), min(y0, min(ys) - 0.05), max(x1, max(xs) + 0.05), \
                    max(y1, max(ys) + 0.05)
        s = min((PANEL_W - 12) / max(x1 - x0, 1e-3), (PANEL_H - 12) / max(y1 - y0, 1e-3))
        ox = (PANEL_W - s * (x1 - x0)) / 2
        oy = (PANEL_H - s * (y1 - y0)) / 2
        px = lambda x, y: (ox + (x - x0) * s, PANEL_H - oy - (y - y0) * s)  # noqa: E731

        for sh in meta.get('shapes', []):
            kind = sh['t']
            if kind == 'line':
                flat = [v for q in sh['pts'] for v in px(*q)]
                if len(flat) >= 4:
                    c.create_line(*flat, fill=sh.get('c', '#444'), width=max(1, sh.get('w', 0.01) * s),
                                  dash=(4, 3) if sh.get('dash') else None)
            elif kind == 'poly':
                flat = [v for q in sh['pts'] for v in px(*q)]
                c.create_polygon(*flat, fill=sh.get('fill', ''), outline=sh.get('c', '#444'))
            elif kind == 'circle':
                (cx, cy), rr = px(*sh['center']), sh['r'] * s
                c.create_oval(cx - rr, cy - rr, cx + rr, cy + rr, fill=sh.get('fill', ''), outline=sh.get('c', '#444'))
            elif kind == 'text':
                c.create_text(*px(*sh['p']), text=sh['s'], fill=sh.get('c', '#444'), font=('TkDefaultFont', 9, 'bold'))

        order = {'ok': 0, 'dead': 1, 'run': 2, 'fail': 3}   # les échecs par-dessus
        shown = ok = fail = 0
        for r in sorted(runs, key=lambda r: order[r.status(done)]):
            st = r.status(done)
            ok += st == 'ok'
            fail += st == 'fail'
            if not self.show.get(r.level, self.show['real']).get():
                continue
            if not self.show['run' if st == 'dead' else st].get():
                continue
            if len(r.pts) < 2:
                continue
            shown += 1
            tag = f'run{id(r)}'
            picked = self.picked is r
            flat = [v for q in r.pts for v in px(*q)]
            c.create_line(*flat, fill=COLORS[st], width=3 if picked else 1.5 if st == 'fail' else 1,
                          dash=(5, 3) if r.level == 'hard' and not picked else None, tags=(tag,))
            ex, ey = px(*r.pts[-1])
            if st == 'fail':
                c.create_line(ex - 4, ey - 4, ex + 4, ey + 4, fill=COLORS[st], width=2, tags=(tag,))
                c.create_line(ex - 4, ey + 4, ex + 4, ey - 4, fill=COLORS[st], width=2, tags=(tag,))
            elif st == 'run':
                c.create_oval(ex - 4, ey - 4, ex + 4, ey + 4, fill=COLORS[st], outline='white', tags=(tag,))
            c.tag_bind(tag, '<Enter>', lambda e, r=r: self.info.config(text=r.label, foreground=COLORS[r.status(
                self.campaign_done())]))
            c.tag_bind(tag, '<Button-1>', lambda e, r=r, key=key: self.pick(r, key))
            if picked:
                c.tag_raise(tag)
        sx, sy = px(*runs[0].pts[0]) if runs[0].pts else (None, None)
        if sx is not None:
            c.create_oval(sx - 3, sy - 3, sx + 3, sy + 3, fill='#222', outline='')
        name = key[0] + (f'  ·  seed {runs[0].seed}' if p['many'] else '')
        p['title'].config(text=f'{name}      {ok}/{len(runs)} réussis' + (f', {fail} échec(s)' if fail else ''),
                          fg=COLORS['fail'] if fail else '#222')

    def pick(self, run, key):
        prev = self.picked
        self.picked = None if prev is run else run
        self.info.config(text=run.label, foreground=COLORS[run.status(self.campaign_done())])
        groups = self.groups()
        for k in {key} | ({prev.key} if prev is not None else set()):
            if k in self.panels:
                self.draw(k, groups.get(k, []))

    # ------------------------------------------------------------------ rafraîchissement
    def tick(self):
        try:
            if self.follow is not None:
                d = self.follow()
                if d is not None and Path(d) != self.dir and Path(d).exists():
                    self.set_dir(Path(d))
            changed, new_group = self.scan()
            if new_group:
                self.rebuild()
            elif changed:
                groups = self.groups()
                for key in changed:
                    if key in self.panels:
                        self.draw(key, groups.get(key, []))
                self.update_count()
        except Exception:
            import traceback
            traceback.print_exc()
        try:
            self.win.after(700, self.tick)
        except Exception:
            pass   # fenêtre fermée


def open_window(root, directory=None, follow=None):
    """Fenêtre secondaire du centre de simulation."""
    import tkinter as tk
    win = tk.Toplevel(root)
    win.title('RobotHub — trajets de toutes les simulations')
    win.geometry('1300x800')
    Viewer(win, directory, follow)
    return win


def main(directory=None):
    import tkinter as tk
    root = tk.Tk()
    root.title('RobotHub — trajets de toutes les simulations')
    root.geometry('1300x800')
    Viewer(root, directory)
    root.mainloop()


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else None)
