# afMF: Low-rank Full Matrix Factorization for dropout imputation in single cell RNA-seq and benchmarking with imputation algorithms for downstream applications 
Install the package locally: (1) download / clone 'afMF' to your directory; (2) enter the 'afMF' directory; (3) run ' pip install . '  <br />
  
To impute the data: <br />
  
import pandas as pd  
from afMF.runafMF import afMF  
  
dat = pd.read_csv("/your_dir/data.txt", sep="\t", index_col=0)  
imputed_dat = afMF(dat)  
print(imputed_dat.iloc[:, : 5].head(5))

Pesudo Code : 

![pesudo_1](https://github.com/user-attachments/assets/463da4ee-3fa7-44df-9524-721b520c9346)
![pesudo_2](https://github.com/user-attachments/assets/e80e6b0f-76b4-48fb-9c6d-1778ac19e881)
![pesudo_3](https://github.com/user-attachments/assets/98099bf7-d06b-4678-b1c7-b28d50a94467)
