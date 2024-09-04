# afMF: Low-rank Full Matrix Factorization for dropout imputation in single cell RNA-seq and benchmarking with imputation algorithms for downstream applications 
Install the package locally: (1) download / clone 'afMF' to your directory; (2) enter the 'afMF' directory; (3) run ' pip install . '  <br />
  
To impute the data: <br />
  
import pandas as pd  
from afMF.runafMF import afMF  
  
dat = pd.read_csv("/your_dir/data.txt", sep="\t", index_col=0)  
imputed_dat = afMF(dat)  
print(imputed_dat.iloc[:, : 5].head(5))

Pesudo Code : 

![pesudo_1](https://github.com/user-attachments/assets/afcd39fc-15b5-4d55-984f-410fa322a878)
![pesudo_2](https://github.com/user-attachments/assets/7ff1af88-3486-48bf-a967-e2fb8bf735ba)
