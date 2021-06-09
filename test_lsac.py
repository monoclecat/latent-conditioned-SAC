import spinup.algos.pytorch.lsac.lsac as lsac
import gym

lsac.lsac(gym.make("Pendulum-v0"), steps_per_epoch=3000, epochs=10)
