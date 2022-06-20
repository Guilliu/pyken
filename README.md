![pyken](https://github.com/guilliu/pyken/blob/main/visual/wallpaper.jpg)
# Pyken

**Pyken** is a Python package with the aim of providing the necessary tools for:
- Grouping variables (both numerical and categorical) in an **automatic and interactive** way.
- Development of **highly customizable scorecards** adaptable to the methodological requirements of each user.

## Installation
You can install Pyken using pip
```
pip install pyken
```

## Template
```python
# Import the modules
import numpy as np, pandas as pd, pyken as pk

# Load the data
from sklearn.datasets import load_breast_cancer as lbc
X, y = pd.DataFrame(lbc().data, columns=lbc().feature_names), lbc().target 

# Apply autoscorecard
model = pk.autoscorecard().fit(X, y)

# Paint the scorecard
pk.pretty_scorecard(model)
```

## Examples
In the folder `/examples` there are notebooks that explore different use cases in more detail.

## Code style
The code tries to be as minimalist as possible. The maximum characters per line is set to 100, since the 80 characters of the PEP 8 standard are considered worse for readability. For all other questions, it is recommended to follow the PEP 8 standards, with a slight preference for the use of single quotes.
