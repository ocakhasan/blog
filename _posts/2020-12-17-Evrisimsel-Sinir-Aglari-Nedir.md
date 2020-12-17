---
layout: post
title:  Evrişimsel Sinir Ağları  (Convolutional Neural Network) Nedir
preview: Derin Öğrenmede resimler üzerinde kullanılan evrişimsel sinir ağlarını kullanıyoruz.
---

Yazıya başlamadan önce belirmek isterim ki, bu tarz derin öğrenme terimlerinin İngilizce ile kullanılması taraftarıyım. Teknik terimlerin Türkçe karşılıkları genelde her zaman duymadığımız kelimeler oluyor ve internette Türkçe pek kaynak yok. Ondan dolayı ben bu terimlerin İngilizce öğrenilip, İngilizce kullanılması taraftarıyım. Herkes global olmaya çalışırken, bizim öyle davranmamamız için hiçbir sebep yok. 

Convolutional sinir ağları genel olarak sıradan sinir ağlarına çok benzerdir. Bu sinir ağları da öğrenebilir ağırlık (weight) ve önyargısı (bias) olan sinirlerden (neuron) oluşur. Her bir nöron bazı inputlar alır, dot product uygular ve bu işlemi lineer olmayan bir yolla devam ettirir. Bütün network hala tek bir ayırt edilebilir skoru açıklar. Network resim pixellerini alıp, sonda bir tahmin üretir. Networkun sonunda belirli  bir kayıp fonksiyonu (loss function) bulunur. 

Peki bu convolutional sinir ağları normal sinir ağlarına bu kadar benziyorsa ne değişiyor? Bu sorunun cevabı ise şu şekildedir: 

Convolutional sinir ağları inputun resimlerden oluştuğunu varsayar, bu varsayım bize bazı özellikleri sisteme entegre etmemize yardımcı olur. 


### YAPISAL GÖZLEM

**Normal Sinir Ağları:** Normal sinir ağları tek bir input alır, onu bazı gizli katmanlardan (hidden layer) geçirir. Her bir hidden layer nöron kümelerinden oluşur,  her bir nöron, bir önceki katmandaki bütün nöronlarla bağlantılıdır ve diğer nöronlardan bağımsız şekilde çalışır. Son katman ise sonuç katmanı (output layer) olarak adlandırılır ve bu katmanda her bir sınıfın olasılığı belli olur. 


Bu normal sinir ağları resimler kullanılınca pek iyi ölçeklenemiyor. Örnek olarak $(32, 32, 3)$ lük boyutlarda resimler kullanırsak, ilk katman $32 * 32 * 3 = 3072$ ağırlığa sahip olacaktır. Bu yük halledilebilir şekilde görülüyor ancak, bu fully-connected yapı büyük resimlere ölçeklenmiyor. Örnek olarak eğer biz boyutları $(200, 200, 3)$ olan resimler kullanırsak, bu sefer nilk nöronlar $200 * 200 * 3 = 120, 000$ ağırlığa sahip olacaklar. Ancak bu büyük numaralı ağırlıklar aşırı uyma (*overfitting*) denilen olaya sebep olacaktır. 


Convolutional sinir ağları ise inputun resimlerden oluşmasınından faydalanır ve buna göre yapıyı daha mantıklı şekilde kurar. Normal sinir ağlarının aksine, Convolutiona sinir ağlarının nöronları 3 boyuta ayarlanmış şekildedir. **genişlik, yükseklik, derinlik**.

Örnek olarak $(32, 32, 3)$ boyutlu resimlerde
* Genişlik = 32
* Yükseklik = 32
* Derinlik = 3
olacaktır. 


### PEKI BU CONVOLUTIONAL SINIR AĞLARI NASIL OLUŞTURULUYOR? 

Bu sinir ağları katman dizilerinden oluşur ve bu katmanlar ise şu şekildedir. 

* Convolutional Katman
* Pooling Katmanı
* Fully-Connected Katmanı

Bu 3 katmandan oluşan katmanları birleştirip bir sinir ağı oluşturacağız. 


#### CONVOLUTIONAL KATMAN

Convolutional katman Convolutional sinir ağlarının büyük ağır işini yapan katmanlardır. 

Conv katmanlar parametreleri öğrenilebilir filtrelerden oluşur. Her bir filtre boyut olarak küçüktür, ancak input derinliği boyunca uzanırlar. Örnek olarak, tipik bir filtre $5 * 5 * 3$ boyutlarında olabilir. İlk 5 genişlik, ikinci 5 yükseklik ve üçüncü 3 ise resimin 3 derinlikli olmasından kaynaklanır. Doğrudan iletme kısmında, her bir filtreyi input resmi üzerinde kaydırıyoruz, bu kaydırma sırasında resimlerde pixeller ile filtredeki sayılar ile dot product alıyoruz. Filtreyi kaydırma işlemi sırasında 2 boyutlu bir aktitive haritası oluşturuyoruz. Bu harita ise bize her bir pozisyondaki cevabı veriyor. Sinir ağı, bu filtreler ne zaman belirli bir görsel özellik, örnek olarak kenar, gördüğü zaman öğrenecek. Her bir filtrenin oluşturduğu haritaları üst üste sıkıştırıp bunu bir sonraki katmana iletiyoruz. 


![Convolutional Sinir Ağları Örnek]({{ site.baseurl }}/images/cnn.png)


**BOYUTSAL AYARLAMA**

Her bir nöronun nasıl bağlı olduğunu anlattık ancak output hacminde kaç tane nöron olduğundan bahsetmedik. Output hacmini belirleyen 3 ayrı parametre vardır. 
* **DERİNLİK:** Bu parametre kaç tane nöron kullandığınıza işaret eder. Örnek olarak ilk convolutional katman input olarak resmi alırken, farklı nöronlar bu resimde farklı detayları fark edebilir. 
* **STRIDE (KAYDIRMA ADIMI):** Bu parametre ise filtreyi kaç pixel kaydıracağımıza işaret eder. Eğer `stride` bir ise, filtreleri bir pixel kaydıracağımız anlamına gelir. 
* **ZERO-PADDING: (SIFIRLARLA DOLDURMA)** Bazı durumlarda inputun etrafını sıfırlarla doldurmak uygun olmaktadır. Bu işlemin güzel bir tarafı ise, bize output boyutunu kontrol altında tutma olanağı vermesidir. Örnek olarak daha yüksek boyutlu outputlar istersek, inputu filtre boyutu kadar sıfırlarla doldurup, bir sonraki katmana aktarılacak outputun boyutunu, şuanki katmandaki input boyutuna eşit tutabiliriz. 


Output hacminin boyutunu şu şekilde hesaplayabiliriz. 
* Input Boyutu = $W$
* Convolutional katman nöronları filtre boyutu = $F$
* Stride = $S$
* Zero-Padding = $P$

Output hacmi boyutu formülü = $(W - F + 2P) / S + 1$. 

Örnek olarak eğer elimizde $10 * 10$ boyutlu bir resim varsa ve bizim filtre boyutumuz $3 * 3$, stride = $1$ ve padding = $0$ ise

$$
Output Boyutu = (10 - 3 + 2*0) / 1 + 1 = 8 * 8 
$$

Şimdi bu boyut tek bir nörondan çıkan sonuç. Eğer elimizde $n$ tane nöron varsa, bu katmandan çıkan sonucun boyutu $8 * 8 * n$ olacaktı. 





