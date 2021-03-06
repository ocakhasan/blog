---
layout: post
title: Flask ve Sklearn ile Film Önerme Sitesi Yapalım 
preview: Metin benzerliği benzerliğini kullanarak Flask ile film önerme sitesi yapalım.
---

**İçerik**
* TOC
{:toc}

Bu yazıdaki bütün kodlar [Bu repodan](https://github.com/ocakhasan/movie-recommender) bulunmaktadır. Eğer demo versiyonunu görmek isterseniz [http://banafilmoner.herokuapp.com/](http://banafilmoner.herokuapp.com/) sitesinden görebilirsiniz.

## Gereksinimler
Bu yazımızda yapacağımız siteyi eğer kendiniz de yapmak istiyorsanız [Flask](https://flask.palletsprojects.com/en/1.1.x/) ve [Scikit-learn](https://scikit-learn.org/) kütüphanelerini yüklemeniz gerekmektedir. Bunları yüklemek için terminalden şu komutları yazabilirsiniz ya da her bir paketin dökümentasyonundan bakabilirsiniz.

```
pip install Flask 
pip install scikit-learn
```

## Sitenin Yapısı
Yapacağımız sitede film önerileri metin benzerliği ile olacak. Bu filmlerin açıklama metinlerini ise bir veri kümesinden alacağız. Bu veri kümesine [TMDB 5000 Movies](https://www.kaggle.com/tmdb/tmdb-movie-metadata) sayfasından ulaşabilirsiniz. Bundan dolayı önerebileceğimiz metinler sadece bu veri kümesindekiler olacaktır. Metin benzerliğini ise [kosinüs benzerliği](https://merveenoyan.medium.com/kosin%C3%BCs-benzerli%C4%9Fi-2b4a4c924f27) ile yapacağız. 

## Veri Seti ve Metin Benzerliği
Veri setindeki `title` filmin başlığını ve `overview` ise filmi basitçe açıklayan metin. Biz `overview` sütununu kullanarak metin benzerliğini kuracağız. Bunun için önce `utils.py` diye bir dosya oluşturalım ve indirdiğimiz veri setini de projedeki dosyaya koyalım. Öncelikle kosinüs benzerliğini verecek olan bir fonksiyon yazalım.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def get_cosine_similarities(df):

    vectorizer = TfidfVectorizer(stop_words="english")

    tf_idf_mat = vectorizer.fit_transform(df['overview'])

    cosine_sim = linear_kernel(tf_idf_mat, tf_idf_mat)

    return cosine_sim
```

`get_cosine_similarities(df)` fonksiyonu parametere olarak `DataFrame` alır, `DataFrame`i ise dosyayı okuduktan sonra alıp daha sonra bu fonksiyona parametre olarak vereceğiz. Fonksiyonda kullanılan [`TfidfVectorizer`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) metinlerden bilgi çıkarmamıza yarayan bir algoritmadır. Açılımı `Term frequency (tf) -> (terim sıklığı)` ve `inverse document frequency (ters döküman sıklığı)`. Yani terimlerin her bir metinde ne kadar sıklıkla geçtiğine ve bir de bütün dökümanda ne kadar sıklıkla geçtiğine bakıp, hangi terimlerin cümleleri ayırmada önemli olduğuna karar verir. Bu bize (4803, n) boyutunda bir matrix dönderecektir. `n` ise bu algoritmanın bulduğu belirleyici kelimelerdir. Yani her bir cümle için her bir kelimenin ne kadar önemi var, bunu gösteren bir matrix. Daha sonra bu matrixi kullanarak her bir metin arasındaki benzerliği bulmak için [`linear_kernel`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.linear_kernel.html) kullanıyoruz. Bu algoritma ise bize (4803, 4803) boyutunda bir matrix dönderecek bu da her bir metinin diğer 4038 filmle benzerliğini gösteren bir matrix olacak. 
Bu fonksiyondan çıkan sonuç ise şu şekildedir

```
[[1.         0.         0.         ... 0.         0.         0.        ]
 [0.         1.         0.         ... 0.02160533 0.         0.        ]
 [0.         0.         1.         ... 0.01488159 0.         0.        ]
 ...
 [0.         0.02160533 0.01488159 ... 1.         0.01609091 0.00701914]
 [0.         0.         0.         ... 0.01609091 1.         0.01171696]
 [0.         0.         0.         ... 0.00701914 0.01171696 1.        ]]
```
Görüldüği gibi bazı değerler 0 bazıları 1 (köşegendekiler), bazıları da 0 ile 1 arasında. Bu demek oluyor ki 0 olanlar arasında hiçbir benzerlik yok, 1 olanlar zaten kendileri ile ölçüldüğü için, örnek olarak 1.film ile 1.film arasındaki benzerlik 1 olacak doğal olarak, 0-1 arasındakiler ise iki film arasındaki benzerliği gösteriyor. 