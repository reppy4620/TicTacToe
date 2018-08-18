from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.properties import ObjectProperty, StringProperty
from kivy.core.window import Window
from kivy.config import Config
from kivy.uix.popup import Popup
from kivy.core.text import LabelBase, DEFAULT_FONT
from kivy.resources import resource_add_path
resource_add_path("/usr/share/fonts/truetype/takao-gothic")
LabelBase.register(DEFAULT_FONT, "TakaoPGothic.ttf")

from Game import Environment
from Agents import get_ddqn

import random as rd
import sys
import time


class MainApp(App):

    def build(self):
        return Gui()


class CellButton(Button):

    def __init__(self, root, num):
        super().__init__()
        self.text = ''
        self.font_size = 100
        self.root = root
        self.num = num

    def on_press(self):
        self.text = '○' if self.root.env.current_player == 1 else 'x'
        self.root.step(self.num)


class YesNoPopUp(BoxLayout):

    text = StringProperty()

    def __init__(self, root, text):
        super(YesNoPopUp, self).__init__(text=text)
        self.root = root

    def yes(self):
        self.root.conte()

    def no(self):
        print('Thank you for playing')
        sys.exit(1)


class Gui(BoxLayout):
    grid = ObjectProperty(None)
    turn = StringProperty()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.env = Environment()
        self.button_list = list()
        self.popup = None
        t = rd.choice([-1, 1])
        self.net = get_ddqn(self.env)
        self.net.load('models/agent2')
        self.net.replay_buffer.load('models/replay2.npz')
        self.turn = 'Your Turn' if t == 1 else 'Enemy Turn'
        self.obs = self.env.reset(t)
        for i in range(3):
            for j in range(3):
                b = CellButton(self, i*3+j)
                self.button_list.append(b)
                self.grid.add_widget(b)
        if t == -1:
            action = self.net.act(self.obs)
            self.step(action)

    def step(self, action):
        if self.env.current_player == 1:
            self.obs, _, game_over, value = self.env.step(action)
            if game_over:
                self.game_over(value)
                return
            action = self.net.act(self.obs)
            self.obs, _, game_over, value = self.env.step(action)
            if game_over:
                self.game_over(value)
            self.button_list[action].text = 'x'
        else:
            self.obs, _, game_over, value = self.env.step(action)
            if game_over:
                self.game_over(value)
            self.turn = 'Your Turn'
            self.button_list[action].text = 'x'

    def game_over(self, value):
        if value == 1:
            text = 'You Win !!'
        elif value == -1:
            text = 'You Lose...'
        elif value == 0:
            text = 'Draw'
        elif value == -2:
            text = 'Enemy Missed'
        else:
            text = 'You Missed'
        self.popup = Popup(title='Game End', content=YesNoPopUp(self, text))
        self.popup.open()

    def conte(self):
        self.popup.dismiss()
        t = rd.choice([-1, 1])
        self.turn = 'Your Turn' if t == 1 else 'Enemy Turn'
        self.obs = self.env.reset(t)
        for button in self.button_list:
            button.text = ''
        if t == -1:
            action = self.net.act(self.obs)
            self.step(action)


if __name__ == '__main__':
    Config.set('input', 'mouse', 'mouse,disable_multitouch')
    Window.size = (1000, 1000)
    MainApp().run()
