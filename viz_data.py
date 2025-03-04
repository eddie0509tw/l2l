from dats.miniimagenet import MiniImagenet
from tools.plot import viz_image

def main():
    mode = 'test'
    mi = MiniImagenet(root='/workspace/datasets', mode=mode, download=True)
    data = mi[1]
    viz_image(data[0], data[1])

if __name__ == '__main__':
    main()