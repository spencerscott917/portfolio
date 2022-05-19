import numpy as np
from ks2d2s import ks2d2s
np.random.seed(1337)

def main():
    x1  = np.random.normal(loc=0, scale=1, size=1000)
    y1  = np.random.normal(loc=0, scale=1, size=1000)
    x2  = np.random.normal(loc=0, scale=1, size=1000)
    y2  = np.random.normal(loc=0, scale=1, size=1000)

    p = ks2d2s(x1, y1, x2, y2)
    print(p)


if __name__ == "__main__":
    main()
