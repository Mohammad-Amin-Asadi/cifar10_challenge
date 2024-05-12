Cifar10 dataset Challenge
![Chat Application](
    https://production-media.paperswithcode.com/datasets/4fdf2b82-2bc3-4f97-ba51-400322b228b1.png
)

 دیتاست Cifat10 با ۶۰ هزار عکس  یکی از دیتاست های محبوب در حوزه بینایی ماشین است که میتوان قدرت مدل ها را در Classification روی عکس های بسیار کوچک (۳۲ × ۳۲) بررسی و مورد آزمایش قرار داد

 هدف از ایجاد این Repo  بررسی قدرت مدل های مختلف از جمله مدل های ساده شبکه عصبی پیچیده (CNN) هاب این دیتاست میباشد

 یکی از موفقیت آمیز ترین مدل ها Resnet بود که تا کنون در مدل های بسیاری از معماری این مدل در سایر مدل های جدید تر استفاده می شوند

 در اینجا یک نمونه مدل ResNet 270K پیاده سازی شده و سعی بر بررسی عملکرد این مدل خواهیم داشت و در پایان میتوان نمودار های مربوطه را به خوبی مشاهده کرد

آپدیت :‌ اطلاعات مربوط به مدل MobileNet نیز اضافه شد ُ لازم به ذکر است که این مدل بهینه سازی برای cifar10 نشده است اما به زودی اطلاعات مدل های کوچکتر اما با دقت شبیه به مدل اصلی در دسترس قرار خواهد گرفت


# Refrences

 Papers

[VGG Paper](https://arxiv.org/pdf/1409.1556.pdf)

[Batch Normalization](https://arxiv.org/pdf/1502.03167.pdf)

[Resnet](https://arxiv.org/pdf/1512.03385.pdf)


# Enviroment setup

 برای راحتی در نصب و اجرای فایل های مربوطه میتوانید به راحتی و فقط با اجرا کردن فایل install.sh محیط لازم برای اجرای کد ها را فراهم آورید

 توجه کنید که برای اجرای فایل install.sh نیاز به اجازه دادن به این فایل جهت اجرا دارید که میتوانید از این دستور استفاده کنید

``` sudo chmod -R 777 .```
 سپس 
``` sudo ./install.sh ```

# How to use

 data
 
   برای دسترسی به این دستاست از tensorflow.keras.datasets در کد استفاده شده و نیازی به دانلود جداگانه ی دیتا های دیتاست ندارید و به محض اجرای کد به صورت خودکار این دیتاست در کد load  خواهد شد

 train

 تنها با اجرا کردن فایل main.py عملیات train به صرت خودکار صورت میگیرد و لاگ Train  در پوشه ای با نام logs قرار خواهد کرفت که به صورت آماده این لاگ ها هم اکنون در این Repo  موجود است و نیازی به Train  برای دسترسی به روند train نخواهید داشت

 Evaluation
برای تست کردن این دیتاست بر روی بخش تست cifar10 فایلی تحت عنوان eval_cifar10.py ساخته شده و با تنها وارد کردن یک آرگومان هنگام اجرا که مسیر فایل .keras و یا .h5 میباشد

 مثال : 
- ``` python3 eval_cifar10.py './model.keras' ```

 convert

 قابلیت convert کردن مدل هنوز پیاده سازی نشده است اما به محض انجام این بخش آپدیت خواهد شد و توضیحات تکمیلی قرار خواهند گرفت


# Evaluation Reports

|  Model      | params  | accuracy | validation loss  |
| :---:       |  :---:  |  :---:   |      :---:       |
| ResNet      |   270 K |    90    |      0.4744      |
| MobileNetV1 | 3.509 M |   90.38  |      0.4625      |


 در این بخش میتوانید نمودار های این مدل را هنگام Train مشاهده نمایید

# Model keras vs onnx vs Quantized onnx formats for MobileNetV1 Benchmarking (Speed , size)

دقت شود که تست سرعت  عددی میانگین است که از تشخیص یک عکس برای صد مرتبه به دست آماده است

|  Model         | size (MB)| speed (ms)    |     accuracy     |
| :---:          |  :---:   |  :---:        |      :---:       |
| keras          |   28.3   |    100        |      90.38       |
| onnx           |   13.9   |     9         |      90.38       |
| Quantized onnx |   3.6    |     24        |      82.92       |



نتیجه ای که گرفتم این بود که میتوان از onnx  برای افزایش سرعت مدل استفاده بسیاری کرد و به این علت که حجم مدل را تا حد بسیار زیادی کم میکند میتوان از این ابزار برای استفاده ی مدل های هوش مصنوعی در سیستم هایی که محدودیت از نظر حافظه و قدرت سخت افزاری دارند استفاده ی بسیاری نمود.


#### Formats speed Plots (NEW)
[evaluation_loss_vs_iteration](/home/mohammad/Desktop/CIFAR10_FS/Gitlab/Report/MobileNetSpeedTest/plots/speed_models.png)



ResNet Plots
[evaluation_loss_vs_iteration](./ResNet/Plots/)

MobileNet Plots
[MobileNet](./MobileNet/plots/)


# Download Models
[DOWNLOAD ResNet Model](https://drive.google.com/file/d/1TAVxMqBrmFTmV4KSTwp52cFEL6f75HKs/view?usp=sharing)
[DOWNLOAD MobileNet Model](https://drive.google.com/file/d/110rINRfM_LCZPNCIpWafLWO7YawMYgeA/view?usp=sharing)"# cifar10_challenge" 
