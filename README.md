# jcmt_transient_alignment
This public repository is the home for the correlation derived alignment methods used in the JCMT-Transient survey. 

## Functions within the module:

### Correlate:
#### IN:
    
  * epoch_1: the first epoch, when epoch_2 is None this makes the function an auto_correlation.
  * epoch_2: the second epoch, when given it is the map which is cross-correlated with epoch 1
  * clipped_side: this is the side of the clipped square from the centre of the map
  * clip_only: True will only clip the map and not correlate it.
  * psd: Generates the Power spectrum which is the Fourier Transform of the Auto-Correlation
  
#### OUT:

  * if epoch_1 is None, the function will raise an exception.
  * if epoch_1 is a 2d array, and epoch_2 is None it will computer the auto-correlation of Epoch_1
  * if Epoch_1 and Epoch_2 are both 2d arrays it will compute the cross-correlation of Epoch 2 to Epoch 1.
  * if psd is True it will compute the power spectrum of Epoch_1 even if epoch_2 is not None. 
  * if clip_only is True, then the map will only be clipped to a clipped_side by clipped_side square array.

### **gaussian_fit_xc**

### **gaussian_fit_ac**

### **fourier_gaussian_function**

### **amp**

### **beam_fit**
