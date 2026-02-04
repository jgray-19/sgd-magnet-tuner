# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/jgray-19/sgd-magnet-tuner/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                              |    Stmts |     Miss |   Cover |   Missing |
|------------------------------------------------------------------ | -------: | -------: | ------: | --------: |
| src/aba\_optimiser/\_\_init\_\_.py                                |        1 |        0 |    100% |           |
| src/aba\_optimiser/accelerators/\_\_init\_\_.py                   |        3 |        0 |    100% |           |
| src/aba\_optimiser/accelerators/base.py                           |       53 |        7 |     87% |63, 123, 133, 158-161, 177 |
| src/aba\_optimiser/accelerators/lhc.py                            |       72 |        0 |    100% |           |
| src/aba\_optimiser/config.py                                      |       55 |        0 |    100% |           |
| src/aba\_optimiser/dataframes/\_\_init\_\_.py                     |        0 |        0 |    100% |           |
| src/aba\_optimiser/dataframes/utils.py                            |       24 |        0 |    100% |           |
| src/aba\_optimiser/dispersion/dispersion\_estimation.py           |      145 |      126 |     13% |39-48, 68-75, 99-129, 142-150, 193-264, 312-345, 361-379, 421-534 |
| src/aba\_optimiser/filtering/\_\_init\_\_.py                      |        1 |        1 |      0% |         8 |
| src/aba\_optimiser/io/\_\_init\_\_.py                             |        1 |        0 |    100% |           |
| src/aba\_optimiser/io/utils.py                                    |       67 |        1 |     99% |        24 |
| src/aba\_optimiser/mad/\_\_init\_\_.py                            |        5 |        0 |    100% |           |
| src/aba\_optimiser/mad/base\_mad\_interface.py                    |      140 |       15 |     89% |101, 204-206, 229, 279, 321-323, 401-403, 411-415, 500 |
| src/aba\_optimiser/mad/dispatch.py                                |        8 |        2 |     75% |     18-19 |
| src/aba\_optimiser/mad/lhc\_optimising\_interface.py              |       29 |        0 |    100% |           |
| src/aba\_optimiser/mad/optimising\_mad\_interface.py              |      169 |       11 |     93% |107, 181-182, 336, 344, 375, 417-425 |
| src/aba\_optimiser/mad/scripts.py                                 |       11 |        0 |    100% |           |
| src/aba\_optimiser/matching/\_\_init\_\_.py                       |        3 |        0 |    100% |           |
| src/aba\_optimiser/matching/matcher.py                            |      299 |      278 |      7% |57-84, 95-223, 237-382, 402-483, 491-496, 504-535, 546-547, 558, 567-576, 594-641, 650-762, 769, 776, 793 |
| src/aba\_optimiser/matching/matcher\_config.py                    |       26 |       10 |     62% |47-54, 62-67 |
| src/aba\_optimiser/measurements/\_\_init\_\_.py                   |        1 |        0 |    100% |           |
| src/aba\_optimiser/measurements/create\_datafile.py               |      353 |      353 |      0% |     1-754 |
| src/aba\_optimiser/measurements/create\_datafile\_b2.py           |       58 |       58 |      0% |     1-113 |
| src/aba\_optimiser/measurements/create\_datafile\_loop.py         |      101 |      101 |      0% |     1-315 |
| src/aba\_optimiser/measurements/knob\_extraction.py               |       23 |       23 |      0% |    17-179 |
| src/aba\_optimiser/measurements/optimise\_closed\_orbit.py        |      231 |      231 |      0% |     1-690 |
| src/aba\_optimiser/measurements/optimise\_squeeze\_quads.py       |      280 |      280 |      0% |     1-814 |
| src/aba\_optimiser/measurements/plot\_quad\_diffs\_and\_phases.py |      249 |      249 |      0% |     1-683 |
| src/aba\_optimiser/measurements/squeeze\_helpers.py               |       67 |       67 |      0% |     7-198 |
| src/aba\_optimiser/measurements/squeeze\_measurements.py          |        8 |        8 |      0% |     4-115 |
| src/aba\_optimiser/measurements/utils.py                          |       71 |       36 |     49% |27-71, 76-80 |
| src/aba\_optimiser/model\_creator/\_\_init\_\_.py                 |        6 |        6 |      0% |      3-15 |
| src/aba\_optimiser/model\_creator/config.py                       |       19 |       19 |      0% |      3-88 |
| src/aba\_optimiser/model\_creator/create\_models.py               |       56 |       56 |      0% |    12-159 |
| src/aba\_optimiser/model\_creator/madng\_utils.py                 |       69 |       69 |      0% |     3-376 |
| src/aba\_optimiser/model\_creator/madx\_utils.py                  |       44 |       44 |      0% |     3-127 |
| src/aba\_optimiser/model\_creator/tfs\_utils.py                   |       26 |       26 |      0% |      3-97 |
| src/aba\_optimiser/optimisers/\_\_init\_\_.py                     |        1 |        0 |    100% |           |
| src/aba\_optimiser/optimisers/adam.py                             |       29 |        0 |    100% |           |
| src/aba\_optimiser/optimisers/amsgrad.py                          |       25 |        0 |    100% |           |
| src/aba\_optimiser/optimisers/lbfgs.py                            |       73 |        1 |     99% |       103 |
| src/aba\_optimiser/physics/\_\_init\_\_.py                        |        1 |        0 |    100% |           |
| src/aba\_optimiser/physics/deltap.py                              |       14 |       14 |      0% |      3-51 |
| src/aba\_optimiser/physics/lhc\_bends.py                          |       30 |        0 |    100% |           |
| src/aba\_optimiser/plotting/\_\_init\_\_.py                       |        1 |        0 |    100% |           |
| src/aba\_optimiser/plotting/strengths.py                          |      253 |       49 |     81% |110, 122-124, 132-136, 147, 184-185, 262-285, 298, 339-342, 392, 431, 454-477, 485, 522-536, 548-566 |
| src/aba\_optimiser/plotting/utils.py                              |       18 |        4 |     78% | 32-35, 45 |
| src/aba\_optimiser/simulation/\_\_init\_\_.py                     |        1 |        0 |    100% |           |
| src/aba\_optimiser/simulation/data\_processing.py                 |      100 |       73 |     27% |41-62, 89-165, 192, 240-251, 265-270 |
| src/aba\_optimiser/simulation/magnet\_perturbations.py            |       55 |       18 |     67% |51, 60-74, 96-99, 105, 113 |
| src/aba\_optimiser/simulation/optics.py                           |       33 |       14 |     58% |36-46, 61-81, 165-175 |
| src/aba\_optimiser/training/\_\_init\_\_.py                       |       13 |        0 |    100% |           |
| src/aba\_optimiser/training/base\_controller.py                   |       36 |        1 |     97% |       149 |
| src/aba\_optimiser/training/configuration\_manager.py             |       62 |        7 |     89% |86-90, 107, 116, 120-122 |
| src/aba\_optimiser/training/controller.py                         |      115 |       20 |     83% |82, 174-182, 196, 247-248, 316, 319-325, 335-336 |
| src/aba\_optimiser/training/controller\_config.py                 |       25 |        0 |    100% |           |
| src/aba\_optimiser/training/controller\_helpers.py                |        5 |        1 |     80% |        38 |
| src/aba\_optimiser/training/data\_manager.py                      |      159 |       27 |     83% |66-68, 77, 113, 134-153, 193, 224, 229-233, 251-252, 286-287, 293, 296-297, 300 |
| src/aba\_optimiser/training/optimisation\_loop.py                 |      163 |       14 |     91% |97, 100-107, 197-198, 224-225, 288-290, 294, 333, 380 |
| src/aba\_optimiser/training/result\_manager.py                    |       61 |        3 |     95% |63-64, 182 |
| src/aba\_optimiser/training/scheduler.py                          |       23 |        1 |     96% |        76 |
| src/aba\_optimiser/training/utils.py                              |       62 |       31 |     50% |33-44, 63, 66, 110-119, 139-154, 178, 185 |
| src/aba\_optimiser/training/worker\_lifecycle.py                  |       54 |       14 |     74% |80-81, 88-91, 106-107, 115-118, 122-123 |
| src/aba\_optimiser/training/worker\_manager.py                    |      199 |       28 |     86% |155, 160, 218, 261, 268, 316, 478-479, 506-530, 534-540, 560-562 |
| src/aba\_optimiser/training\_optics/\_\_init\_\_.py               |        2 |        0 |    100% |           |
| src/aba\_optimiser/training\_optics/controller.py                 |      139 |       19 |     86% |189, 215, 217, 292-298, 337, 362-366, 369-372, 384, 426-429 |
| src/aba\_optimiser/workers/\_\_init\_\_.py                        |        6 |        0 |    100% |           |
| src/aba\_optimiser/workers/abstract\_worker.py                    |       77 |       41 |     47% |97, 109, 118, 137, 149, 170-220, 229, 242-270, 282 |
| src/aba\_optimiser/workers/common.py                              |       84 |        4 |     95% |164, 180, 205, 253 |
| src/aba\_optimiser/workers/optics.py                              |      102 |       54 |     47% |145, 157-163, 180-200, 208-218, 238-248, 262-294, 320-350, 362 |
| src/aba\_optimiser/workers/tracking.py                            |      175 |      101 |     42% |108, 134, 139, 222-230, 241-250, 266-272, 289, 309-332, 344-369, 395-423, 432-498 |
| src/aba\_optimiser/workers/tracking\_position\_only.py            |       82 |       57 |     30% |70-89, 93-95, 101-113, 120-125, 136-148, 152-157, 161-181 |
| **TOTAL**                                                         | **5052** | **2673** | **47%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/jgray-19/sgd-magnet-tuner/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/jgray-19/sgd-magnet-tuner/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/jgray-19/sgd-magnet-tuner/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/jgray-19/sgd-magnet-tuner/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fjgray-19%2Fsgd-magnet-tuner%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/jgray-19/sgd-magnet-tuner/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.