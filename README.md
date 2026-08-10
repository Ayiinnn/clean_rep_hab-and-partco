## 文件组成

#### 1. hab

精简整理后的A+B+尺度控制，目前其中两个训练文件均能做到和我当初实验一致

#### 2. partco 

partco原文件，仅在几处加了一两行关键问题的修补

实测数据: || simgcd: 71.5 || partco_no_part: 75.6 || partco: 78.4 || paper: 81.1 ||

partco 本身无确定性，两次运行结果都会不同

#### 3. diagnosis 

可忽略，正在利用这些干净项目文件做一些诊断实验，尝试除直接缩小unsup尺度外的解决方法

首先确认无part分支的hab和pco标签acc几乎无差异，后期hab更高。hab标签显著尖锐，gate前pair acc差异不显著。基于此调整门控。

调整门控之后通过率几乎无差异，gate后pair acc差异仍有但不再显著。但仍然存在断崖等问题，遂展开进一步实验，列出问题清单。以下问题清单基于原始pco和调整门控后的原始尺度hab。

问题清单(恢复原始尺度1的诊断结果):
1. Always，hab产生pair的结构的效果都稍差 (标签差不多甚至更准，但pair acc会稍低: 错误更容易造成异类同标签，有更强聚集性)，未追因。
   
2. part分支在30 ep之前，为pco带来了标签准确率的细微提升，hab几乎0作用(相对无part版)，未追因。
   
3. (主)hab 在30ep处，grad(main)#gA,主分支#都接近，平均grad(part unsup)/grad(main)即gU/gA与pco接近，但是一个epoch中part unsup的激活频率低(这个有可能是偶然)，而单次unsup激活产生的grad(part unsup)是pco三倍(这个在30ep几个batch，以及后面临近几个epoch都这样), 猜测是断崖成因。定位到Hard negative项在pco和hab的gU中占比98%以上，怀疑HN温度失配，正在实验。

-单独调高hard negative温度30ep断崖消失80%左右，但是后几个ep ACC彻底崩了(依旧大幅震荡，之前版本也是，但只关注断崖了)，仍在追因，
-在解决了断崖之后，目前来看，不论是A+B还是修复断崖后的A+B还是原版partco，part unsup gPos尖峰/gHN尖峰/高active rate等等叠加pair低质量均能导致瞬时低谷。 partco本身尖峰少，且尖峰能显著影响conf(关门), 且pair acc始终高(不知是原因还是结果还是既是原因又是结果)，因此遇到这些虽下降但很轻微。仍在追因。

4. (主)31ep-35ep hab标签准确率pair准确率等等等等全崩了，pco gU/gA 降低至1/3(虽然大后期也会涨上来) hab gU/gA 没降。问题较多，这个留到3.解决后再追因，因为很可能是断崖后果。
