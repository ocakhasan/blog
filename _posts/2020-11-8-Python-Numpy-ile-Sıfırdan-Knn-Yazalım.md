---
layout: post
title: Python Numpy ile Sıfırdan K Nearest Neighbours Algoritmasını Yazalım 
---

Merhaba bu yazımızda Makine Öğrenmesinde meşhur bir algoritma olan Knn algoritmasını sıfırdan yazacağız. Tabii ki
hazır bir sürü kütüphane var ancak sıfırdan algoritmayı yazabilmek bize algoritmanın nasıl çalışacağını gösterecektir. 
Böylece Knn algoritması bir tahmin yaparken nasıl yapıyor olayın arkasında neler dönüyor bunları anlayabiliyor olacağız. 

## K-Nearest Neighbour Nedir

Öncelikle şunu bilmek gerekir ki K-Nearest-Neighbour adından da anlaşılacağı üzere en yakın k komşu noktalara bakıp en çok hangi label varsa o labelı tahmin(prediction)  olarak verir. 

Peki bu yakınlık uzaklık ilişkisi nasıl kurulur önce ona bakalım. Uzaklığı ölçebilmek için belli başlı algoritmalar vardır. Bunlardan biri `eucledian` diğeri de `manhattan` uzaklığıdır. 

### Eucledian Uzaklığı 

Manhattan uzaklığında aslında iki nokta arasında uzaklığı alırken normal 2 boyutlu denklemde nasıl alıyorsak, bunun n boyutlu formüle döndürülmüş halidir. 

Örnek olarak $a = (x_1, y_1)$ ve $b = (x_2, y_2)$ olsun. Bu noktalar arasında uzaklığı bulurken yaptığımız işlem $$d(a, b) = \sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2}$$

Peki eğer bizim verimiz n boyutlu olursa bu uzaklık nasıl ölçülecek?. Bu durumda ise uzaklık $$d(a,b)= \sum_{i=1}^n (a_i - b_i)^2$$

Bu formulu ise Numpy ile şu şekilde yazabiliriz

```python
np.sqrt(np.sum(np.square(a - b), axis=1))
```

### Manhattan Uzaklığı 

Manhattan uzaklığında iki nokta arasındaki uzaklık her bir alt noktanın farkının mutlak değerlerinin toplamı ile bulunur. 


Örnek olarak $a = (x_1, y_1)$ ve $b = (x_2, y_2)$ olsun. Bu noktalar arasında uzaklığı bulurken yaptığımız işlem $$d(a, b) = \lvert x_1 - x_2\rvert + \lvert y_1 - y_2 \rvert$$


Peki eğer bizim verimiz n boyutlu olursa bu uzaklık nasıl ölçülecek?. Bu durumda ise uzaklık $$d(a,b)= \sum_{i=1}^n \lvert a_i - b_i\rvert$$

Bu formulu ise Numpy ile şu şekilde yazabiliriz
```python
np.sum(np.abs(a - b), axis=1)
```
## Algoritma Akışı

KNN algoritmasında eğitme (training) işlemi aslında sadece verilen veriyi ezberlemekten ibarettir. Bundan dolayı eğitme kısmında bir şey yapmayacağız ancak tahmin etme (prediction) kısmında ise asıl üstteki formuülleri kullanıp işlem yapacağız. 

Bu algoritmayı Python ile Numpy  Kullanarak implement edeceğiz. 
Önceklikle şu komut ile Numpy kütüphanesini import edelim.
```







