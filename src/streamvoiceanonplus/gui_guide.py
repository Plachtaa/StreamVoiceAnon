"""The desktop tutorial with the highlighted controls and step cards."""

from dataclasses import dataclass


@dataclass(frozen=True)
class GuideStep:
    target: object
    title: str
    text: str
    tab: str | None = None


class VisualGuide:
    def __init__(self, root, tabs, ctk, steps):
        import tkinter as tk

        self.root, self.tabs, self.ctk, self.steps = root, tabs, ctk, steps
        self.active = False
        self.index = 0
        self.edges = [tk.Frame(root, borderwidth=0, highlightthickness=0) for _ in range(4)]
        self._build()
        root.bind('<Configure>', self._resize, add='+')
        root.bind('<Escape>', self._escape, add='+')
        root.bind('<Left>', self._key_move(-1), add='+')
        root.bind('<Right>', self._key_move(1), add='+')

    def _build(self):
        ctk = self.ctk
        accent = ('#77500b', '#f0c040')
        self.card = ctk.CTkFrame(self.root, width=298, corner_radius=12, border_width=2,
                                 fg_color=('#fff9ec', '#1e1e32'), border_color=('#a57416', '#f0c040'))
        self.counter = ctk.CTkLabel(self.card, text='', text_color=accent, font=ctk.CTkFont(size=12, weight='bold'))
        self.counter.pack(anchor='w', padx=16, pady=(12, 0))
        self.title = ctk.CTkLabel(self.card, text='', anchor='w', font=ctk.CTkFont(size=17, weight='bold'))
        self.title.pack(fill='x', padx=16, pady=(0, 4))
        self.description = ctk.CTkLabel(self.card, text='', anchor='w', justify='left', wraplength=260,
                                         font=ctk.CTkFont(size=13))
        self.description.pack(fill='x', padx=16, pady=(0, 7))

        progress = ctk.CTkFrame(self.card, fg_color='transparent')
        progress.pack(padx=12, pady=(0, 6))
        self.dots = []
        for index in range(len(self.steps)):
            dot = ctk.CTkButton(progress, text='○', width=23, height=24, fg_color='transparent',
                                text_color=accent, hover_color=('#f4e5bb', '#3b3450'),
                                command=lambda index=index: self.show(index))
            dot.pack(side='left', padx=1)
            self.dots.append(dot)

        buttons = ctk.CTkFrame(self.card, fg_color='transparent')
        buttons.pack(fill='x', padx=16, pady=(0, 13))
        ctk.CTkButton(buttons, text='Skip', width=48, height=29, fg_color='transparent',
                      text_color=('#544321', '#d1cddd'), hover_color=('#f4e5bb', '#3b3450'),
                      command=self.close).pack(side='left')
        self.back = ctk.CTkButton(buttons, text='Back', width=59, height=29, fg_color=('#a58038', '#3b3450'),
                                  command=lambda: self.move(-1))
        self.next = ctk.CTkButton(buttons, text='Next', width=65, height=29, fg_color='#f0c040',
                                  hover_color='#d2a331', text_color='#271a0b', command=lambda: self.move(1))
        self.next.pack(side='right')
        self.back.pack(side='right', padx=(0, 6))

    def start(self):
        self.active = True
        self.root.focus_set()
        self.show(0)

    def refresh(self):
        if self.active:
            self._place()

    def tab_changed(self):
        if self.active and self.steps[self.index].tab not in (None, self.tabs.get()):
            self.close()

    def show(self, index):
        self.index = index
        step = self.steps[index]
        if step.tab and step.tab != self.tabs.get():
            # Use the same synchronous switch as a click. CTkTabview.set queues
            # a delayed hide that can hide the current tab during quick navigation.
            self.tabs._segmented_button.set(step.tab, from_button_callback=True)
        self.counter.configure(text=f'Step {index + 1} of {len(self.steps)}')
        self.title.configure(text=step.title)
        self.description.configure(text=step.text)
        self.back.configure(state='normal' if index else 'disabled')
        self.next.configure(text='Done' if index == len(self.steps) - 1 else 'Next')
        for i, dot in enumerate(self.dots):
            dot.configure(text='●' if i == index else '○')
        self.root.update_idletasks()
        self._place()

    def move(self, direction):
        target = self.index + direction
        if target == len(self.steps):
            self.close()
        elif target >= 0:
            self.show(target)

    def close(self):
        self.active = False
        self.card.place_forget()
        for edge in self.edges:
            edge.place_forget()

    def _escape(self, _event):
        if self.active:
            self.close()
            return 'break'

    def _key_move(self, direction):
        def handler(_event):
            if self.active:
                self.move(direction)
                return 'break'

        return handler

    def _resize(self, event):
        if event.widget is self.root:
            self.refresh()

    def _place(self):
        widget = self.steps[self.index].target
        if not widget.winfo_ismapped():
            for edge in self.edges:
                edge.place_forget()
            return
        x = widget.winfo_rootx() - self.root.winfo_rootx()
        y = widget.winfo_rooty() - self.root.winfo_rooty()
        width, height = widget.winfo_width(), widget.winfo_height()
        near_x, near_y = x - 4, y - 4
        for edge, bounds in zip(self.edges, (
            (near_x, near_y, width + 8, 2), (near_x, y + height + 2, width + 8, 2),
            (near_x, near_y, 2, height + 8), (x + width + 2, near_y, 2, height + 8),
        )):
            edge.configure(bg='#f0c040' if self.ctk.get_appearance_mode() == 'Dark' else '#a57416')
            edge.place(x=bounds[0], y=bounds[1], width=bounds[2], height=bounds[3])
            edge.lift()

        card_w, card_h = self.card.winfo_reqwidth(), self.card.winfo_reqheight()
        root_w, root_h = self.root.winfo_width(), self.root.winfo_height()
        gap = 14
        align_x = max(8, min(x, root_w - card_w - 8))
        align_y = max(8, min(y, root_h - card_h - 8))
        positions = [(x + width + gap, align_y), (x - card_w - gap, align_y),
                     (align_x, y + height + gap), (align_x, y - card_h - gap)]
        card_x, card_y = next(((cx, cy) for cx, cy in positions
                               if 8 <= cx <= root_w - card_w - 8 and 8 <= cy <= root_h - card_h - 8),
                              (align_x, align_y))
        scale = self.card._get_widget_scaling()
        self.card.place(x=card_x / scale, y=card_y / scale)
        self.card.lift()
