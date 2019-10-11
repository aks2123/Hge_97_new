

```python
from fastai import *
from fastai.vision import *
from fastai.core import *
```


```python

```


```python
path=Path('data/')
```


```python
path.ls()
```




    [PosixPath('data/test'),
     PosixPath('data/valid'),
     PosixPath('data/train'),
     PosixPath('data/rawdata'),
     PosixPath('data/hge_97.pkl'),
     PosixPath('data/models')]




```python

```


```python

```


```python
data=ImageDataBunch.from_folder (path,train='train',valid='valid', ds_tfms=get_transforms (), size=224)
```


```python
data.normalize(imagenet_stats)
```




    ImageDataBunch;
    
    Train: LabelList (1474 items)
    x: ImageList
    Image (3, 224, 224),Image (3, 224, 224),Image (3, 224, 224),Image (3, 224, 224),Image (3, 224, 224)
    y: CategoryList
    no_hge,no_hge,no_hge,no_hge,no_hge
    Path: data;
    
    Valid: LabelList (631 items)
    x: ImageList
    Image (3, 224, 224),Image (3, 224, 224),Image (3, 224, 224),Image (3, 224, 224),Image (3, 224, 224)
    y: CategoryList
    no_hge,no_hge,no_hge,no_hge,no_hge
    Path: data;
    
    Test: None




```python

```


```python
data.show_batch(rows=3, figsize=(7,6))
```


![png](output_9_0.png)



```python
print(data.classes)
len(data.classes),data.c
```

    ['hge', 'no_hge']





    (2, 2)




```python
learn = cnn_learner(data, models.resnet50, metrics= accuracy)
```


```python
learn.fit_one_cycle(10)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.369724</td>
      <td>0.370431</td>
      <td>0.863708</td>
      <td>00:13</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.221543</td>
      <td>0.294532</td>
      <td>0.912837</td>
      <td>00:10</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.172452</td>
      <td>0.133616</td>
      <td>0.942948</td>
      <td>00:11</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.131591</td>
      <td>0.143132</td>
      <td>0.939778</td>
      <td>00:10</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.109851</td>
      <td>0.101930</td>
      <td>0.955626</td>
      <td>00:11</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.092997</td>
      <td>0.123203</td>
      <td>0.944533</td>
      <td>00:10</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.074439</td>
      <td>0.113661</td>
      <td>0.942948</td>
      <td>00:10</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.067848</td>
      <td>0.090288</td>
      <td>0.955626</td>
      <td>00:10</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.059481</td>
      <td>0.094406</td>
      <td>0.955626</td>
      <td>00:11</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.052431</td>
      <td>0.088769</td>
      <td>0.955626</td>
      <td>00:10</td>
    </tr>
  </tbody>
</table>



```python
learn.save('hge_asnr2_model')
```


```python
interp = ClassificationInterpretation.from_learner(learn)
```


```python
losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)
```




    True




```python
interp.plot_top_losses(9, figsize=(15,11))
```


![png](output_16_0.png)



```python
interp.plot_confusion_matrix(figsize=(4,4), dpi=60)
```


![png](output_17_0.png)



```python
learn.unfreeze()
```


```python
learn.fit_one_cycle(1)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.121481</td>
      <td>0.083958</td>
      <td>0.965135</td>
      <td>00:13</td>
    </tr>
  </tbody>
</table>



```python
learn.load('hge_asnr2_model');
```


```python
learn.lr_find()
```





    LR Finder is complete, type {learner_name}.recorder.plot() to see the graph.



```python
learn.recorder.plot()
```


![png](output_22_0.png)



```python
learn.unfreeze()
learn.fit_one_cycle(3, max_lr=slice(1e-4,1e-6))
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.020554</td>
      <td>0.150671</td>
      <td>0.950872</td>
      <td>00:12</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.018580</td>
      <td>0.096419</td>
      <td>0.960380</td>
      <td>00:12</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.018617</td>
      <td>0.067737</td>
      <td>0.974643</td>
      <td>00:12</td>
    </tr>
  </tbody>
</table>



```python
learn.export('hge_97_new.pkl')
```


```python
learn.save('hge_model_97_new')
```


```python
interp = ClassificationInterpretation.from_learner(learn)
```


```python

```


```python
interp.plot_top_losses(9, figsize=(15,11))
```


![png](output_28_0.png)



```python
interp.plot_confusion_matrix(figsize=(4,4), dpi=60)
```


![png](output_29_0.png)



```python
preds,y=learn.get_preds()
```


```python
im=open_image('data/test/new12.tiff')
```


```python
pred_class, pred_idx,outputs=learn.predict(im)
```


```python
pred_class
```




    Category no_hge




```python
im.show()
```


![png](output_34_0.png)



```python

```
