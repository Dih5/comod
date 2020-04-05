# World population data
This data was obtained from the following reference:
United Nations, Department of Economic and Social Affairs, Population Division (2019). World Population Prospects 2019, Online Edition. Rev. 1.,
which is available at https://population.un.org/wpp/Download/Standard/CSV/ under a [Creative Commons license CC BY 3.0 IGO](http://creativecommons.org/licenses/by/3.0/igo/).

The following snippet was used to create the file.
```python
import pandas as pd
df=pd.read_csv("https://population.un.org/wpp/Download/Files/1_Indicators%20(Standard)/CSV_FILES/WPP2019_TotalPopulationBySex.csv")
df[df["Time"]==2019][["Location", "PopTotal"]].to_csv("population.csv", index=False)
```
Note it takes some time to download the file.
