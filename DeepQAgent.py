import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys


import random
import time


from utils import screenshot, image_to_tensor, show_img

class Game:
    def __init__(self, custom_config=True):
        chrome_options = Options()
        chrome_options.add_argument('--mute-audio')
        self.browser = webdriver.Chrome(executable_path=chromebrowser_path, options=chrome_options)
        self.browser.set_window_position(x=-10, y=0)
        self.browser.get('chrome://dino')
        self.browser.execute_script("document.getElementsByClassName('runner-canvas')[0].id = 'runner-canvas'")
        self.browser.execute_script("Runner.config.ACCELERATION=0")
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

        is_over = False  # game over

        if actions[0] == 1:
            self._agent.jump()
            reward = -5
        elif actions[1] == 1:
            self._agent.duck()
            reward = -3
        elif actions[2] == 1:
            self._agent.DoNothing()
            reward = 1

        image = screenshot(self.dinoGame.browser)
        self._display.send(image)

        if self._agent.is_crashed():
            # generation_score.append(score)
            # time.sleep(0.1)
            self.dinoGame.restart()
            reward = -100
            is_over = True

        image = image_to_tensor(image)
        return image, reward, is_over, score, high_score


class ConvNN(nn.Module):

    def __init__(self):
        super(ConvNN, self).__init__()

        self.number_of_actions = 3
        self.gamma = 0.99
        self.final_epsilon = 0.0001
        self.initial_epsilon = 0.1
        self.number_of_iterations = 5000000
        self.replay_memory_size = 60000
        self.minibatch_size = 64

        self.conv1 = nn.Conv2d(4,32,8,4,2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 4, 2,1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64,64,3,1,1)
        self.maxpool = nn.MaxPool2d(2)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc4 = nn.Linear(64, 32)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc5 = nn.Linear(32, self.number_of_actions)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.maxpool(x)
        x = self.relu3(x)
        x = x.view(x.size()[0], -1)
        x = self.fc4(x)
        x = self.relu4(x)
        out = self.fc5(x)

        return out

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.uniform(m.weight, -0.01, 0.01)
        m.bias.data.fill_(0.01)

def train(model, start):
    optimizer = optim.Adam(model.parameters(), lr=0.0004)
    criterion = nn.MSELoss()

    game = Game()
    dino = DinoAgent(game)
    game_state = Game_state(dino, game)

    replay_memory = []

    action = torch.zeros([model.number_of_actions], dtype=torch.float32)
    action[0] = 1
    image_data, reward, terminal, score, high_score = game_state.get_next_state(action)

    state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)

    print(f'printing size of input state at 0: \n {state.size()}')

    epsilon = model.initial_epsilon
    iteration = 0

    epsilon_decrements = np.linspace(model.initial_epsilon, model.final_epsilon, model.number_of_iterations)

    while iteration < model.number_of_iterations:
        output = model(state)[0]

        action = torch.zeros([model.number_of_actions], dtype=torch.float32)
        if torch.cuda.is_available():
            action = action.cuda()

        action_index = [torch.argmax(output)][0]

        if torch.cuda.is_available():
            action_index = action_index.cuda()

        action[action_index] = 1

        image_data_1, reward, terminal, score, high_score = game_state.get_next_state(action)
        state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)

        action = action.unsqueeze(0)
        reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0)

        replay_memory.append((state, action, reward, state_1, terminal))

        if len(replay_memory) > model.replay_memory_size:
            replay_memory.pop(0)

        epsilon = epsilon_decrements[iteration]
        minibatch = random.sample(replay_memory, min(len(replay_memory), model.minibatch_size))

        state_batch = torch.cat(tuple(d[0] for d in minibatch))
        action_batch = torch.cat(tuple(d[1] for d in minibatch))
        reward_batch = torch.cat(tuple(d[2] for d in minibatch))
        state_1_batch = torch.cat(tuple(d[3] for d in minibatch))

        if torch.cuda.is_available():
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            state_1_batch = state_1_batch.cuda()

        output_1_batch = model(state_1_batch)

        y_batch = torch.cat(tuple(reward_batch[i] if minibatch[i][4]
                                  else reward_batch[i] + model.gamma * torch.max(output_1_batch[i])
                                  for i in range(len(minibatch))))

        q_value = torch.sum(model(state_batch) * action_batch, dim=1)
        optimizer.zero_grad()

        y_batch = y_batch.detach()
        loss = criterion(q_value, y_batch)

        loss.backward()
        optimizer.step()

        state = state_1

        # global generation_score
        # if len(generation_score) == 0:
        #     avg_score = 0
        # else:
        #     avg_score = sum(generation_score)/len(generation_score)

        if iteration % 10000 == 0:
            print(f'iteration: {iteration}')
            torch.save(model, 'pretrained-model/current_model_' + str(iteration) + '.pth')

        iteration += 1

def DeepQAgent(model):

    game = Game()
    dino = DinoAgent()
    game_state = Game_state(dino, game)

    action = torch.zeros([model.number_of_actions], dtype=torch.float32)
    action[0] = 1
    image_data, reward, terminal, s_, h_ = game_state.get_next_state(action)
    state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)

    while True:
        output = model(state)[0]

        action = torch.zeros([model.number_of_actions], dtype=torch.float32)
        if torch.cuda.is_available():
            action = action.cuda()

        action_index = torch.argmax(output)
        if torch.cuda.is_available():
            action_index = action_index.cuda()
        action[action_index] = 1

        image_data_1, reward, terminal, s_, h_ = game_state.get_next_state(action)
        state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)

        state = state_1
        print(f'average generation score: {mean(generation_score)}')

def main(mode):
    cuda_is_available = torch.cuda.is_available()

    if mode == 'test':

        model = torch.load(
            'pretrained-model/current_model_200000.pth',
            map_location='cpu' if not cuda_is_avaiable else None
        ).eval()

        if cuda_is_available:
            model = model.cuda()

        test(model)

    elif mode =='train':
        if not os.path.exists('pretrained-model/'):
            os.mkdir('pretrained-model/')

        model = ConvNN()

        if cuda_is_available:
            model = model.cuda()

        model.apply(init_weights)
        start = time.time()

        train(model, start)


generation_score = []

game_url = "chrome://dino"
chromebrowser_path = "/home/snowballfight/Documents/chromedriver"


if __name__ == '__main__':
    mode = 'train'
    main(mode)
    print(mode)











