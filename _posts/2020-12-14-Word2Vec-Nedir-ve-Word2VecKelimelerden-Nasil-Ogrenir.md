---
layout: post
title: Word2Vec Nedir ve Word2Vec Kelimelerden Nasıl Öğrenir
preview: Natural Language Processing'de kullanılan Word2Vec modelini inceliyoruz.
---

Makine öğrenmesinde modellerin veriyi görme şekli biz insanlardan farklıdır. Biz kolayca ***Kırmızı arabayı görüyorum.*** cümlesini anlayabilirken, model bu kelimeleri anlayacak vektörlere ihtiyaç duyar. Bu vektörlere `word embeddings` denir. 

### WORD VECTORLERİ NASIL ÇALIŞIR - Tablodan Bak

Her kelimemiz için belirli bir boyutta vektörümüz olacak ve bu vektörleri kelimeyi isteyerek alabiliriz.

Buna key-value pair örneği verilebilir. 
* key: kelime 
* value: vektör

Bundan dolayı herhangi bir kelimenin vektörüne bakmak için dictionaryden kelimeyi istediğimiz zaman vektöre ulaşmış olacağız.


## Word2Vec: Tahmin Bazlı bir Metod

Ana amacımız kelimelerden, kelime vektörleri oluşturmak. 

Word2Vec parametreli word vektörleri olan bir modeldir. Bu parametreler itaretive yöntemle, objective function(küçültmeye çalıştığımız fonksiyon) kullanarak optimize edilir. 

Peki bunu nasıl yapacağız. 

Unutmadan:
* amaç : her bir vektörü kelimenin içeriğini bilecek şekilde kodlamak
* nasıl yapılacak: vektörleri kelimelerden olası içerik tahmin edecek şekilde eğitmek.


`Word2Vec` iterative bir metottur. Ana fikirleri kısaca şöyledir. 
* büyük bir text corpusu alır
* texti,  belirli bir sliding window(kayan pencere) kullanarak, her seferinde bir kelime ilerleyecek şekilde ilerlemek. Her bir adımda, bir tane `central word (merkezi kelime)` ve `context words(içerik kelimeleri)` -> penceredeki diğer kelimeler. 
* merkezi kelime için, içerik kelimelerinin olasılıklarını hesapla.
* vektörleri olasılıkları artıracak şekilde ayarla

![word2vec-nedir]({{ site.baseurl }}/images/training_data.png)

Resimde de görüleceği üzere her seferinde arkası mavi olan `merkezi kelime` ve diğerleri de `içerik kelimeleri`. 


### Objective Function (Amaç Fonksiyonu)

Text corpusundaki her bir $ t = 1, ... , T$ pozisyon için, Word2Vec merkezi kelimesi $w_{t}$ verilmiş m-boyutlu penceredeki içerik kelimelerini tahmin eder. 

$$
Likelihood = L(\theta) = \prod_{t=1}^{T} \prod_{-m \leq j \leq m, j \neq 0}  P(w_{t + j} \mid w_t, \theta) 
$$

Bu fonksiyonda $\theta$ optimize edilecek bütün parametrelerdir. Amaç ve Kayıp Fonksiyonu $J(\theta) ise ortalama negatif log olabilirlik fonksiyonudur. (Negative log-likelihood)

$$
J(\theta) = -\frac{1}{T} \log L(\theta) = -\frac{1}{T} \sum_{t=1}^{T} \sum_{-m \leq j \leq m, j \neq 0 } \log P(w_{t + j} \mid w_t, \theta)
$$


Bu formüldeki parçalara ayıralım.
* $\sum_{t=1}^{T}$ Bu kısım bütün text üzerinde gezinir. 
* $\prod_{-m \leq j \leq m, y \neq 0}$ bu ise kayma penceresini(sliding window) temsil eder.
* $\log P(w_{t + j} \mid w_t, \theta)$ : bu ise merkezi kelimesi verilen içeriğin olasılığını hesaplar.

Peki asıl sorulması gereken soru bu olasılıklar nasıl hesaplanacak?

### Olasılıkları Nasıl Hesaplayacağız?
Hesaplamak istediğimiz olasılık 
$$
P(w_{t + j} \mid w_t, \theta)
$$

Verilen her kelime $w$ için, iki adet vektörümüz var.

* $v_w$ -> kelimenin merkezi kelime (central word) olduğu zaman
* $u_w$ -> kelimenin içerik kelime (context word) olduğu zaman


Vektörler train edildikten sonra, genel olarak içerik vektörlerini $u_w$ atar ve sadece merkezi kelime vektörlerini $v_w$ kullanılır.

Bundan sonra verilen `merkezi kelime` $c$ ve `içerik kelimesi` $o$ kelimeleri için olasılık: 

$$
P(o \mid c) = \frac{exp(u_{o}^{T})}{\sum_{v \in V} exp(u_{w}^{T} v_c)}
$$

**NOT:** Bu bir `softmax fonksiyonudur. ` Softmax ile alakalı yazıma [bu yazımdan](https://ocakhasan.github.io/blog/Softmax-Aktivasyon-Fonksiyonu-Nedir-Numpy-Implementasyonu/) ulaşabilirsiniz.



