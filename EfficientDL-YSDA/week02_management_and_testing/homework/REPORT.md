# week 2 homework

## what i did

i started by running all tests

```bash
python -m pytest -x -vv
```

this command stopped on the first error

so i fixed one problem and ran it again

## unet bug

the first error was inside unet

the time embedding had shape `[b c]`

the image features had shape `[b c h w]`

i added two empty axes

```python
temb = temb[:, :, None, None]
```

now the shape is `[b c 1 1]`

these axes do not add new values

they let pytorch add the same time value to every point of the feature map

## diffusion bugs

the diffusion test gave a wrong loss

i checked how noise was created

the code used uniform noise

ddpm needs normal noise

so i changed this

```python
eps = torch.randn_like(x)
```

the noisy image also used the wrong value from the schedule

i changed it to `sqrt_one_minus_alpha_prod`

after this the diffusion test passed

## test changes

i added a fixture in `tests/conftest.py`

it resets the random seed before every test

this makes test results stable

my mac has no cuda

so cuda tests are skipped when cuda is not available

they can still run on a gpu machine

the original `test_training` was empty

i made a small full pipeline test

it uses four cifar10 images

it uses a small unet

it uses only two diffusion steps

this keeps the test fast

with learning rate `0` the weights must stay the same

with learning rate `0.001` some weights must change

the test also generates an image

then it checks that the png file exists

## test result

```text
8 passed
2 skipped
```

the two skipped tests need cuda

coverage for `modeling/training.py` is

```text
100%
```

## hydra

before this all settings were written inside `main.py`

i moved them into yaml configs

```text
conf
  config.yaml
  optimizer
    adam.yaml
    sgd.yaml
  wandb
    default.yaml
```

adam is used by default

sgd can be selected from the terminal

the config also stores batch size

it stores number of epochs

it stores number of workers

it stores random flip settings

it also stores model and diffusion settings

training is off by default

this lets me check configs without starting a long run

## wandb

i added wandb settings to the config

main sends the full hydra config to wandb

main can also send generated images

source code upload is disabled

i only made a basic wandb setup

loss logging is not added yet

learning rate logging is not added yet

input logging is not added yet

config artifact logging is not added yet

i did not run 100 epochs

so there is no final wandb link

## how to run

show the final config

```bash
python main.py --cfg job
```

check the config without training

```bash
python main.py
```

check sgd config

```bash
python main.py optimizer=sgd --cfg job
```

change some values

```bash
python main.py \
  optimizer=sgd \
  optimizer.momentum=0.8 \
  training.batch_size=64 \
  training.num_epochs=3 \
  data.random_flip=true \
  --cfg job
```

start training

```bash
python main.py training.enabled=true
```

run without online wandb

```bash
python main.py \
  training.enabled=true \
  wandb.mode=offline
```

## what is left

log training loss

log learning rate

log one input batch

save config as a wandb artifact

run full experiments

add a wandb link

i did not add dvc
