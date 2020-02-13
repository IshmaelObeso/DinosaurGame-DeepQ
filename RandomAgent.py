import torch
import time

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys

from utils import screenshot, image_to_tensor, show_img

class Game:
    def __init__(self, custom_config=True):
        chrome_options = Options()
        chrome_options.add_argument('--mute-audio')
        self.browser = webdriver.Chrome(executable_path=chromebrowser_path, options=chrome_options)
        self.browser.set_window_position(x=-10, y=0)
        self.browser.get('chrome://dino')
        self.browser.execute_script("document.getElementsByClassName('runner-canvas')[0].id = 'runner-canvas'")
        self.browser.implicitly_wait(30)
        self.browser.maximize_window()

    def get_crashed(self):
        return self.browser.execute_script('return Runner.instance_.crashed')

    def get_playing(self):
        return self.browser.execute_script('return Runner.instance_.playing')

    def restart(self):
        self.browser.execute_script('Runner.instance_.restart')

    def press_up(self):
        self.browser.find_element_by_tag_name('body').send_keys(Keys.ARROW_UP)

    def press_down(self):
        self.browser.find_element_by_tag_name('body').send_keys(Keys.ARROW_DOWN)

    def press_right(self):
        self.browser.find_element_by_tag_name('body').send_keys(Keys.ARROW_RIGHT)

    def get_score(self):
        score_array = self.browser.execute_script('return Runner.instance_.distanceMeter.digits')
        score = ''.join(score_array)
        return int(score)

    def get_highscore(self):
        score_array = self.browser.execute_script('return Runner.instance_.distanceMeter.highScore')
        for i in range(len(score_array)):
            if score_array[i] == '':
                break
        score_array = score_array[i:]
        score = ''.join(score_array)
        return int(score)

    def pause(self):
        return self.browser.execute_script('return Runner.instance_.stop()')

    def resume(self):
        return self.browser.execute_script('return Runner.instance_.play()')

    def end(self):
        self.browser.close()


class DinoAgent:
    def __init__(self, game):
        self.dinoGame = game
        self.jump()

    def is_running(self):
        return self.dinoGame.get_playing()

    def is_crashed(self):
        return self.dinoGame.get_crashed()

    def jump(self):
        self.dinoGame.press_up()

    def duck(self):
        self.dinoGame.press_down()

    def DoNothing(self):
        self.dinoGame.press_right()


class Game_state:
    def __init__(self, agent, game):
        self._agent = agent
        self.dinoGame = game
        self._display = show_img()
        self._display.__next__()

    def get_next_state(self, actions):

        score = self.dinoGame.get_score()
        high_score = self.dinoGame.get_highscore()

        reward = 0.1
        is_over = False  # game over

        if actions[0] == 1:
            self._agent.jump()
        elif actions[1] == 1:
            self._agent.duck()
        elif actions[2] == 1:
            self._agent.DoNothing()

        image = screenshot(self.dinoGame.browser)
        self._display.send(image)

        if self._agent.is_crashed():
            generation_score.append(score)
            time.sleep(0.1)
            self.dinoGame.restart()
            reward = -1
            is_over = True

        image = image_to_tensor(image)
        return image, reward, is_over, score, high_score


def RandomAgent():
    game = Game()
    dino = DinoAgent(game)
    game_state = Game_state(dino, game)
    number_of_actions = 3

    action = torch.zeros([number_of_actions], dtype=torch.float32)
    action[0] = 1

    image_data, reward, terminal, s_, h_ = game_state.get_next_state(action)
    state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)

    while True:
        action = torch.zeros([number_of_actions], dtype=torch.float32)
        action_index = [torch.randint(number_of_actions, torch.Size([]), dtype=torch.int)]
        action[action_index] = 1
        image_data_1, reward, terminal, s_, h_ = game_state.get_next_state(action)
        print(f'image data: {image_data_1}, {image_data_1.shape}')
        state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)
        state = state_1






generation_score = []

game_url = "chrome://dino"
chromebrowser_path = "/home/snowballfight/Documents/chromedriver"

dino_1 = RandomAgent()