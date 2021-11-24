# Linear multiple regression totally from scratch
Author: Minegishi Gouki
## Prepare requirement  
##### tested only ubuntu18.04 and python=3.8.12
```
pip install numpy
pip install pandas
```
## Data  
##### [Auto-mpg.csv](https://archive.ics.uci.edu/ml/datasets/Auto+MPG)
- target columns : mpg
- features : other than mpg   

Auto-mpg have 8 features as below  
This heatmap means Correlation coefficient
![EDA](EDA.png)

## Implement Linear Regression  
```
python regression.py -f feature1 -f feature2 ....
```
You can choose features below
- "weight"
- "horsepower"
- "cylinders"
- "displacement""
- "acceleration"
- "model year"
- "origin"

Each features must be str type