# DM_underwater
This is the code of the paper "Underwater Image Enhancement by Transformer-based Diffusion Model with Non-uniform Sampling for Skip Strategy"

usage steps:

- Install necessary Python packages from requirement.txt.
- Putting your data into the dataset folder. (There is initial data in this folder now).
- Download the pre-trained model, the link is https://drive.google.com/file/d/1As3Pd8W6XmQBU__83iYtBT5vssoZHSqn/view?usp=sharing. Then, put the model in the experiments_supervised folder.
- Execute infer.py to get the inference results in a new folder called experiments.
- Users can also comment and uncomment the line 13 and 14 to change for the training process. And execute train.py for training.
- search_diffussion.py is used to search the sequence of time steps with the evolutionary algorithm. Users can use it in the inference process.

P.S. The author is so lazy that he doesn't want to write down more instructions.


