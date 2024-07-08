from cube import Cube
from constants import *
from utility import *

import random
import random
import numpy as np



class Snake:
    body = []
    turns = {}
    it = 0
    s_dis = 0
    def __init__(self, color, pos, file_name=None):
        # pos is given as coordinates on the grid ex (1,5)
        self.color = color
        self.head = Cube(pos, color=color)
        self.body.append(self.head)
        self.dirnx = 0
        self.dirny = 1
        self.it = 0
        try:
            self.q_table = np.load(file_name)
        except:
            self.q_table = np.zeros((4096, 4))

        self.lr = 0.095 # TODO: Learning rate
        self.discount_factor = 0.95 # TODO: Discount factor
        self.epsilon = 0 # TODO: Epsilon

    def get_self_dir(self, state):
        return state // 4**5
    
    def get_self_around(self, state):
        return (state//4 ** 2) % (4**3)
    
    def get_snack_dir(self, state):
        return (state % 16)//4

    def get_snack_len(self, state):
        return state % 4

    def get_optimal_policy(self, state):
        opt = 0

        for i in range(4):
            if (i == 5 - self.get_self_dir(state) or i == 1 - self.get_self_dir(state)):
                continue
            if (self.q_table[state, i] >= self.q_table[state, opt]):
                opt = i
        return opt

    def make_action(self, state):
        chance = random.random()
        if chance < self.epsilon:
            action = random.randint(0, 3)
            while (1):
                dir = self.calc_dir()
                if (dir + action == 1 or dir + action == 5):
                    action = random.randint(0, 3)
                else: 
                    break
        else:
            action = self.get_optimal_policy(state)
        return action

    def update_q_table(self, state, action, next_state, reward):
        sample = reward
        mx = 0
        for i in range(4):
            mx = max(self.q_table[next_state, i], mx)
        sample += self.discount_factor * mx
        self.q_table[state, action] = (1 - self.lr) * self.q_table[state, action] + self.lr * sample
        return 

    def calc_dir(self):
        if (self.dirnx == 1):
            return 1
        if (self.dirnx == -1):
            return 0
        if (self.dirny == 1):
            return 3
        return 2
    
    def calc_snack_dir(self, snack):
        x = snack.pos[0]
        y = snack.pos[1]
        s_x = self.head.pos[0]
        s_y = self.head.pos[1]
        dir = self.calc_dir()
        if (x <= s_x and y <= s_y):
            if (dir == 3):
                return 0
            if (dir == 1):
                return 2
            if (abs(s_y - y) > abs(s_x - x)):
                return 2
            return 0
        if (x <= s_x and y >= s_y):
            if (dir == 2):
                return 0
            if (dir == 1):
                return 3
            if (abs(s_y - y) > abs(s_x - x)):
                return 3
            return 0
        
        if (x >= s_x and y <= s_y):
            if (dir == 3):
                return 1
            if (dir == 0):
                return 2
            if (abs(s_y - y) > abs(s_x - x)):
                return 2
            return 1
        if (dir == 2):
            return 1
        if (dir == 0):
            return 3
        if (abs(s_y - y) > abs(s_x - x)):
            return 3
        return 1

    def create_state(self, snack, other_snake):
        self_around = self.calc_around(other_snake)
        dir = self.calc_dir()
        le = self_around % 4**dir
        ri = self_around // 4 ** (dir + 1)
        self_around = ri * (4 ** dir) + le
        snack_dir = self.calc_snack_dir(snack)
        snack_len = min(4, self.dis(snack))
        return int(dir * 4 ** 5 + self_around * 4 ** 2 +  snack_dir * 4 + snack_len - 1)

    def dis(self, snack):
        return abs(snack.pos[0] - self.head.pos[0]) + abs(snack.pos[1] - self.head.pos[1])

    def calc_around(self, other_snake):
        x = self.head.pos[0]
        y = self.head.pos[1]
        mnr = 20 - x
        mnl = x + 1
        mnd = 20 - y
        mnu = y + 1
        for j in other_snake.body:
            i = j.pos
            if (i[0] == x):
                if (i[1] > y):
                    mnd = min(mnd, abs(i[1] - y))
                else:
                    mnu = min(mnu, abs(i[1] - y))
            if (i[1] == y):
                if (i[0] > x):
                    mnr = min(mnr, abs(i[0] - x))
                else:
                    mnl = min(mnl, abs(i[0] - x))
        for j in self.body:
            i = j.pos
            if (i[0] == x and i[1] == y):
                continue
            if (i[0] == x):
                if (i[1] > y):
                    mnd = min(mnd, abs(i[1] - y))
                else:
                    mnu = min(mnu, abs(i[1] - y))
            if (i[1] == y):
                if (i[0] > x):
                    mnr = min(mnr, abs(i[0] - x))
                else:
                    mnl = min(mnl, abs(i[0] - x))
        sum = 0
        mnu = min(4, mnu)
        mnd = min(4, mnd)
        mnr = min(4, mnr)
        mnl = min(4, mnl)
        sum += 4 ** 2 * (mnu - 1)
        sum += 4 ** 3 * (mnd - 1)
        sum += 4 ** 0 * (mnl - 1)
        sum += 4 ** 1 * (mnr - 1)
        return sum

    def move(self, snack, other_snake):
        state = self.create_state(snack, other_snake)
        action = self.make_action(state)
        self.it += 1
        self.s_dis = self.dis(snack)
        if (self.it % 100 == 0):
            #self.epsilon *= 0.9
            self.lr *= 0.9
            self.lr = max(0.045, self.lr)
           # self.epsilon = max(0.1, self.epsilon)

        if action == 0: # Left
            self.dirnx = -1
            self.dirny = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
        elif action == 1: # Right
            self.dirnx = 1
            self.dirny = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
        elif action == 2: # Up
            self.dirny = -1
            self.dirnx = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
        elif action == 3: # Down
            self.dirny = 1
            self.dirnx = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]

        for i, c in enumerate(self.body):
            p = c.pos[:]
            if p in self.turns:
                turn = self.turns[p]
                c.move(turn[0], turn[1])
                if i == len(self.body) - 1:
                    self.turns.pop(p)
            else:
                c.move(c.dirnx, c.dirny)

        new_state = self.create_state(snack, other_snake)
        return state, new_state, action
        # TODO: Create new state after moving and other needed values and return them
    
    def check_out_of_board(self):
        headPos = self.head.pos
        if headPos[0] >= ROWS - 1 or headPos[0] < 1 or headPos[1] >= ROWS - 1 or headPos[1] < 1:
            self.reset((random.randint(3, 18), random.randint(3, 18)))
            return True
        return False
    
    def calc_reward(self, snack, other_snake):
        reward = 0
        is_near = self.dis(snack) - self.s_dis

        win_self, win_other = False, False
        if (self.dis(snack) < 3):
            if (is_near < 0):
                reward += 2000
            else:
                reward -= 3000
        else:
            if (is_near < 0):
                reward = +2000
            if (is_near > 0):
                reward = -10000

        self_around = self.calc_around(other_snake)
        dir = self.calc_dir()
        le = self_around % 4**dir
        ri = self_around // 4 ** (dir + 1)
        self_around = ri * (4 ** dir) + le
        for i in range(3):
            x = self_around % 4
            self_around //= 4
            if (x == 0):
                reward -= 3000
            if (x == 1):
                reward -= 1000
            if (x == 2):
                reward += 20
            else:
                reward += 50   
        
        if self.check_out_of_board():
            # TODO: Punish the snake for getting out of the board
            win_other = True
            reward -= 40000
           
            reset(self, other_snake)
            
        
        if self.head.pos == snack.pos:
            self.addCube()
            snack = Cube(randomSnack(ROWS, self), color=(0, 255, 0))
            # TODO: Reward the snake for eating
            reward += 10000

        if self.head.pos in list(map(lambda z: z.pos, self.body[1:])):
            # TODO: Punish the snake for hitting itself
            reward -= 40000
            win_other = True
            reset(self, other_snake)
            
            
        if self.head.pos in list(map(lambda z: z.pos, other_snake.body)):
            
            if self.head.pos != other_snake.head.pos:
                # TODO: Punish the snake for hitting the other snake
                reward -= 40000
                win_other = True
            else:
                if len(self.body) > len(other_snake.body):
                    # TODO: Reward the snake for hitting the head of the other snake and being longer
                    reward += 40000
                    win_self = True
                elif len(self.body) == len(other_snake.body):
                    # TODO: No winner
                    reward += 2000
                else:
                    reward -= 40000
                    # TODO: Punish the snake for hitting the head of the other snake and being shorter
                    win_other = True


            reset(self, other_snake)
            
        return snack, reward, win_self, win_other
    
    def reset(self, pos):
        self.head = Cube(pos, color=self.color)
        self.body = []
        self.body.append(self.head)
        self.turns = {}
        self.dirnx = 0
        self.dirny = 1

    def addCube(self):
        tail = self.body[-1]
        dx, dy = tail.dirnx, tail.dirny

        if dx == 1 and dy == 0:
            self.body.append(Cube((tail.pos[0] - 1, tail.pos[1]), color=self.color))
        elif dx == -1 and dy == 0:
            self.body.append(Cube((tail.pos[0] + 1, tail.pos[1]), color=self.color))
        elif dx == 0 and dy == 1:
            self.body.append(Cube((tail.pos[0], tail.pos[1] - 1), color=self.color))
        elif dx == 0 and dy == -1:
            self.body.append(Cube((tail.pos[0], tail.pos[1] + 1), color=self.color))

        self.body[-1].dirnx = dx
        self.body[-1].dirny = dy

    def draw(self, surface):
        for i, c in enumerate(self.body):
            if i == 0:
                c.draw(surface, True)
            else:
                c.draw(surface)

    def save_q_table(self, file_name):
        np.save(file_name, self.q_table)
        