import os
import PIL
import PIL.GifImagePlugin
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math

class Space:
    def __init__(self, wall:np.ndarray, reflactive_index:np.ndarray) -> None:
        self.wall = wall
        self.reflactive_index = reflactive_index

class WaveClass:
    def __init__(self, amplitude:float, wavelength:float, frequency: float, E0:np.ndarray, E1:np.ndarray) -> None:
        self.amplitude = amplitude  # 振幅
        self.wavelegth = wavelength # 波長
        self.frequency = frequency  # 周波数
        self.waves = [E0, E1]       # 波の状態(0: t-dt, 1: t)
        
    # 速度
    @property
    def velocity(self) -> float:
        return self.wavelegth * self.frequency
    
    # 波数
    @property
    def wavenumber(self) -> float:
        return 2 * math.pi / self.wavelegth
    
    # 周波数
    @property
    def period(self) -> float:
        return 1 / self.frequency    
    
    # 角周波数
    @property
    def angular_frequency(self) -> float:
        return 2 * math.pi * self.frequency

    # 波の更新
    def update(self, space: Space) -> None:
        next = np.zeros([size, size])
        for x in range(0, size):
            for y in range(0, size):
                # 境界条件 => 境界で吸収
                if x == 0 or x == size-1 or y == 0 or y == size-1:
                    if (x==0 and y==0):
                        next[x][y] = self.waves[0][x+1][y] + self.waves[0][x][y+1]
                    elif (x==0 and y==size-1):
                        next[x][y] = self.waves[0][x+1][y] + self.waves[0][x][y-1]
                    elif (x==size-1 and y==0):
                        next[x][y] = self.waves[0][x-1][y] + self.waves[0][x][y+1]
                    elif (x==size-1 and y==size-1):
                        next[x][y] = self.waves[0][x-1][y] + self.waves[0][x][y-1]
                    elif x == 0:
                        next[x][y] = self.waves[0][x+1][y]
                    elif x == size-1:
                        next[x][y] = self.waves[0][x-1][y]
                    elif y == 0:
                        next[x][y] = self.waves[0][x][y+1]
                    elif y == size-1:
                        next[x][y] = self.waves[0][x][y-1]
                # 設置された壁で反射
                elif space.wall[x][y] == 1:
                    next[x][y] = 0
                # 波動方程式の計算
                else:
                    next[x][y] = 2 * self.waves[1][x][y] - self.waves[0][x][y] + (self.velocity/space.reflactive_index[x][y])**2 * k*((self.waves[1][x+1][y] - 2*self.waves[1][x][y] + self.waves[1][x-1][y]) + (self.waves[1][x][y+1] - 2*self.waves[1][x][y] + self.waves[1][x][y-1]))
        
        self.waves[0] = self.waves[1]
        self.waves[1] = next*alpha        
        
        
# 画面表示に関わる設定
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot()
ax.set_aspect("equal")

# シミュレートするサイズ
size = 100
time = 1000

# シミュレーションの分解能
L = size*10 # 区間
N = size # 空間分割数
T = time*100 # 時間分割数
dt = time/T
dx = L/N
k=(dt/dx)**2

# 増幅係数
alpha = 1

# 壁
wall = np.zeros([size, size])
wall[size//4+5][:] = 1
wall[size//4+5][size//2-10] = 0
wall[size//4+5][size//2+10] = 0

# 屈折率
reflactive_index = np.ones([size, size])
reflactive_index[size//4:size][:] = 1.5
reflactive_index[size//4+5:size][:] = 1

# シミュレーションの初期化
space = Space(wall=wall, reflactive_index=reflactive_index)
wave = WaveClass(amplitude=size*size/100, wavelength=dx, frequency=dt*1000, E0=np.zeros([size, size]), E1=np.zeros([size, size]))
t = 0

# 初期条件
wave.waves[1][1][size//2] = wave.amplitude*math.sin(wave.wavenumber*size//2*dx - wave.angular_frequency*dt)

# メイン関数
def main():
    make_gif_abs("wave_abs.gif")


# リアルタイムで正負を考慮した強度分布を表示
def draw(wave: np.ndarray):
    global fig, ax
    ax.clear()
    ax.pcolor(wave, cmap="coolwarm", vmin=-1, vmax=1)
    plt.pause(0.01)

# リアルタイムで強度分布を表示
def draw_abs(wave: np.ndarray):
    global fig, ax
    ax.clear()
    ax.pcolor(np.abs(wave), cmap="gray", vmin=0, vmax=1)
    plt.pause(0.01)
   
# 正負を考慮した強度分布のgifを作成(メモリを多く使うので注意) 
def make_gif(filename: str):
    global fig, ax
    print("Make coolwarm gif")
    ims = []
    for i in range(0, time):
        im = ax.pcolor(wave.waves[1], cmap="coolwarm", vmin=-1, vmax=1)
        ims.append([im])
        wave.update(space=space)
        print(format(i/time*100, ".2f") + "%")
    del im, i
    print("100.00%")
    print("rendering...")
    ani = animation.ArtistAnimation(fig, ims, interval=100)
    ani.save(filename="./wave/"+filename, writer="pillow")
    print("complete")
    del ims, ani
       
# 強度分布のgifを作成(メモリを多く使うので注意)
def make_gif_abs(filename: str):
    global fig, ax
    print("Make gray gif")
    ims = []
    for i in range(0, time):
        im = ax.pcolor(np.abs(wave.waves[1]), cmap="gray", vmin=0, vmax=1)
        ims.append([im])
        wave.update(space=space)
        print(format(i/time*100, ".2f") + "%")
    del im, i
    print("100.00%")
    print("rendering...")
    ani = animation.ArtistAnimation(fig, ims, interval=100)
    ani.save(filename="./wave_abs/"+filename, writer="pillow")
    print("complete")
    del ims, ani

# coolwarmとgrayの両方を表示(メモリを特に多く使うので注意)
def make_gif_both(filename: str):
    global fig, ax
    print("Make both gif")
    ims = []
    for i in range(0, time):
        im1 = ax.pcolor(wave.waves[1], cmap="coolwarm", vmin=-1, vmax=1)
        im2 = ax.pcolor(np.abs(wave.waves[1]), cmap="gray", vmin=0, vmax=1)
        ims.append([im1, im2])
        wave.update(space=space)
        print(format(i/time*100, ".2f") + "%")
    del im1, im2, i
    print("100.00%")
    print("rendering...")
    ani = animation.ArtistAnimation(fig, ims, interval=100)
    ani.save(filename=filename, writer="pillow")
    print("complete")
    del ims, ani

if __name__=="__main__":
    main()