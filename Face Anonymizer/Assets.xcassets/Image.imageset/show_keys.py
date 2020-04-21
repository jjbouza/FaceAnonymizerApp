import matplotlib.pyplot as plt
import matplotlib.image as mpimg

key_out = [(802.1928007964494, 346.64413511521155), (821.2853576422665, 318.27273832684824), (778.909622370501, 324.1152514743434), (854.4616936709631, 332.0044847777845), (721.1994334417558, 343.659135989178), (912.6217853234435, 475.52765723796205), (690.2095809520914, 489.0630699781128), (945.2017342534047, 663.5624867005107), (666.2847021674368, 682.0206579067363), (945.4394151264592, 806.4671616609922), (638.3645618008269, 820.958095209144), (862.0527571741245, 808.8351357307879), (714.155453930569, 810.1978584022373), (899.4288819309338, 1111.2112794868676), (729.2789206894455, 1091.8890214615758), (951.2888155094844, 1330.7384446437256), (758.6390157617948, 1315.4944750121595)]


nose = key_out[0:1]
right_eye = key_out[1:2]
left_eye = key_out[2:3]
right_ear = key_out[3:4]
left_ear = key_out[4:5]
right_shoulder = key_out[5:6]
left_shoulder = key_out[6:7]


x = [key[0] for key in key_out]
y = [key[1] for key in key_out]

img = mpimg.imread('./person-standing.jpg')
plt.imshow(img)
plt.scatter(x, y, c='r')
plt.show()
