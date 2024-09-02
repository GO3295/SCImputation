# afMF: Low-rank Full Matrix Factorization for dropout imputation in single cell RNA-seq and benchmarking with imputation algorithms for downstream applications 
Install the package locally: (1) download / clone 'afMF' to your directory; (2) enter the 'afMF' directory; (3) run ' pip install . '  <br />
  
To impute the data: <br />
  
import pandas as pd  
from afMF.runafMF import afMF  
  
dat = pd.read_csv("/your_dir/data.txt", sep="\t", index_col=0)  
imputed_dat = afMF(dat)  
print(imputed_dat.iloc[:, : 5].head(5))
