# 对热力图的分析

![heatmap](../2.MLR-Multiple_Linear_Regression/figures/img.png)

- 由图可知几个信息
  - Medina,HouseAge,AveRooms与我们设置的因变量MedHouseVal相关性较强(都为正的)
  - AveRooms与AveBedrms强相关

因此可知,**Medina,HouseAge,AveRooms**这三个变量是**关键变量**,也称为输入变量

接下来就要利用这三个变量去训练数据