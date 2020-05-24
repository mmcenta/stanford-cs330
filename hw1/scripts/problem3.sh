# problem 3 runs and plot
python hw1.py --num_classes=2 --num_samples=1 --name="p3" --label="1-shot, 2-way"
python hw1.py --num_classes=3 --num_samples=1 --name="p3" --label="1-shot, 3-way"
python hw1.py --num_classes=4 --num_samples=1 --name="p3" --label="1-shot, 4-way"
python hw1.py --num_classes=4 --num_samples=5 --name="p3" --label="5-shot, 4-way"
python plot.py --group-prefix p3