import numpy as np

def add_noise(n, noise_level):
    noise = np.random.normal(0, noise_level, (n, n))
    return noise




def main():
    print(add_noise(3, .001))


if __name__ == "__main__":
    main()

