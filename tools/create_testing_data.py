import os
import shutil

images_38 = ['000302115.jpg', '000001621.jpg', '000254562.jpg', '000092800.jpg', '000072159.jpg', '000139620.jpg', '000260954.jpg', '000090440.jpg', '000129733.jpg', '000036916.jpg', '000232432.jpg', '000230222.jpg', '000202742.jpg', '000123748.jpg', '000205183.jpg', '000278172.jpg', '000214408.jpg', '000285355.jpg', '000114133.jpg', '000207794.jpg', '000024455.jpg', '000267960.jpg', '000245542.jpg', '000241897.jpg', '000036286.jpg', '000069767.jpg', '000095520.jpg', '000275541.jpg', '000208060.jpg', '000240561.jpg', '000221078.jpg', '000231091.jpg', '000207906.jpg', '000104918.jpg', '000042817.jpg', '000141007.jpg', '000075846.jpg', '000231443.jpg', '000229090.jpg', '000062961.jpg', '000135907.jpg', '000134751.jpg', '000108235.jpg', '000042284.jpg', '000045680.jpg', '000259072.jpg', '000063268.jpg', '000214977.jpg', '000087408.jpg', '000115339.jpg', '000211769.jpg', '000136235.jpg', '000187604.jpg', '000218512.jpg', '000195623.jpg', '000042569.jpg', '000195290.jpg', '000302542.jpg', '000274751.jpg', '000180914.jpg', '000063839.jpg', '000088578.jpg', '000043963.jpg', '000011013.jpg', '000111166.jpg', '000261373.jpg', '000251018.jpg', '000249028.jpg', '000266983.jpg', '000028041.jpg', '000008701.jpg', '000191568.jpg', '000208344.jpg', '000206615.jpg', '000270773.jpg', '000287631.jpg', '000117488.jpg', '000104724.jpg', '000279601.jpg', '000114528.jpg', '000017954.jpg', '000076990.jpg', '000290413.jpg', '000175103.jpg', '000294573.jpg', '000050667.jpg', '000219514.jpg', '000237671.jpg', '000173286.jpg', '000184438.jpg', '000009786.jpg', '000025321.jpg', '000120623.jpg', '000088145.jpg', '000139885.jpg', '000000439.jpg', '000080138.jpg', '000105652.jpg', '000134629.jpg', '000114534.jpg', '000220920.jpg', '000301552.jpg', '000146632.jpg', '000201503.jpg', '000042308.jpg', '000270963.jpg', '000054680.jpg', '000072013.jpg', '000090050.jpg', '000133559.jpg', '000123699.jpg', '000286616.jpg', '000211684.jpg', '000253491.jpg', '000156049.jpg', '000013859.jpg', '000023164.jpg', '000161080.jpg', '000257870.jpg', '000110789.jpg', '000166703.jpg', '000038968.jpg', '000219325.jpg', '000256968.jpg', '000131935.jpg', '000244809.jpg', '000301689.jpg', '000036215.jpg', '000128301.jpg', '000292926.jpg', '000172681.jpg', '000274487.jpg', '000220521.jpg', '000184448.jpg', '000230048.jpg', '000286030.jpg', '000136422.jpg', '000071487.jpg', '000062418.jpg', '000141870.jpg', '000160110.jpg', '000201288.jpg', '000005827.jpg', '000036341.jpg', '000155214.jpg', '000143796.jpg', '000092378.jpg', '000274116.jpg', '000198708.jpg', '000184672.jpg', '000052168.jpg', '000141625.jpg', '000291560.jpg', '000120764.jpg', '000277577.jpg', '000277890.jpg', '000006607.jpg', '000238204.jpg', '000041119.jpg', '000131835.jpg', '000194777.jpg', '000047813.jpg', '000043705.jpg', '000176590.jpg', '000130333.jpg', '000270666.jpg', '000150952.jpg', '000056326.jpg', '000090596.jpg', '000290794.jpg', '000243823.jpg', '000249946.jpg', '000258956.jpg', '000215726.jpg', '000191303.jpg', '000190552.jpg', '000209666.jpg', '000286342.jpg', '000141378.jpg', '000094624.jpg', '000134015.jpg', '000241402.jpg', '000126917.jpg', '000224441.jpg', '000103367.jpg', '000111643.jpg', '000205244.jpg', '000298767.jpg', '000167963.jpg', '000290182.jpg', '000051204.jpg', '000067796.jpg', '000044561.jpg', '000269143.jpg', '000169468.jpg', '000052912.jpg', '000136701.jpg', '000294973.jpg', '000191667.jpg', '000291480.jpg', '000243176.jpg', '000168594.jpg', '000070062.jpg', '000191590.jpg', '000169215.jpg', '000280415.jpg', '000253716.jpg', '000239571.jpg', '000006837.jpg', '000167149.jpg', '000085202.jpg', '000229306.jpg', '000048617.jpg', '000224859.jpg', '000285359.jpg', '000184783.jpg', '000015800.jpg', '000051016.jpg', '000076562.jpg', '000082564.jpg', '000271784.jpg', '000023909.jpg', '000225514.jpg', '000167572.jpg', '000074095.jpg', '000238812.jpg', '000044783.jpg', '000165946.jpg', '000299878.jpg', '000151129.jpg', '000183691.jpg', '000251500.jpg', '000034263.jpg', '000063477.jpg']
images_43 = ['000196995.jpg', '000206053.jpg', '000294247.jpg', '000269787.jpg', '000257949.jpg', '000113034.jpg', '000130417.jpg', '000268572.jpg', '000283198.jpg', '000003832.jpg', '000140292.jpg', '000051814.jpg', '000002560.jpg', '000040476.jpg', '000219996.jpg', '000099885.jpg', '000013669.jpg', '000233638.jpg', '000001655.jpg', '000290204.jpg', '000198997.jpg', '000056517.jpg', '000210909.jpg', '000260831.jpg', '000023952.jpg', '000285772.jpg', '000277669.jpg', '000039181.jpg', '000169668.jpg', '000091888.jpg', '000031493.jpg', '000005033.jpg', '000301523.jpg', '000131718.jpg', '000159712.jpg', '000192345.jpg', '000253643.jpg', '000258500.jpg', '000220301.jpg', '000009568.jpg', '000149038.jpg', '000153527.jpg', '000281526.jpg', '000008588.jpg', '000139037.jpg', '000262581.jpg', '000136351.jpg', '000257968.jpg', '000286595.jpg', '000276433.jpg', '000283127.jpg', '000194310.jpg', '000270573.jpg', '000094747.jpg', '000177298.jpg', '000242392.jpg', '000173567.jpg', '000270047.jpg', '000057479.jpg', '000137599.jpg', '000195189.jpg', '000247773.jpg', '000233852.jpg', '000006304.jpg', '000221583.jpg', '000065724.jpg', '000056914.jpg', '000194497.jpg', '000099958.jpg', '000179367.jpg', '000235411.jpg', '000233741.jpg', '000162156.jpg', '000138312.jpg', '000220110.jpg', '000140289.jpg', '000271712.jpg', '000121173.jpg', '000235227.jpg', '000163439.jpg', '000274398.jpg', '000222595.jpg', '000290721.jpg', '000115380.jpg', '000007807.jpg', '000069605.jpg', '000181426.jpg', '000283653.jpg', '000261220.jpg', '000124961.jpg', '000243706.jpg', '000178788.jpg', '000141914.jpg', '000208250.jpg', '000257416.jpg', '000232560.jpg', '000146982.jpg', '000033952.jpg', '000011481.jpg', '000082575.jpg', '000276453.jpg', '000198732.jpg', '000037074.jpg', '000297406.jpg', '000266813.jpg', '000186201.jpg', '000216317.jpg', '000023282.jpg', '000111645.jpg', '000247659.jpg', '000121684.jpg', '000104305.jpg', '000053038.jpg', '000257227.jpg', '000252608.jpg', '000282455.jpg', '000195487.jpg', '000069995.jpg', '000277797.jpg', '000253049.jpg', '000112416.jpg', '000078807.jpg', '000231311.jpg', '000102350.jpg', '000259453.jpg', '000154029.jpg', '000227904.jpg', '000204185.jpg', '000094064.jpg', '000183817.jpg', '000073096.jpg', '000151539.jpg', '000177552.jpg', '000251106.jpg', '000117846.jpg', '000043575.jpg', '000123919.jpg', '000169454.jpg', '000196822.jpg', '000095108.jpg', '000080850.jpg', '000128440.jpg', '000108308.jpg', '000136547.jpg', '000046994.jpg', '000170002.jpg', '000120596.jpg', '000295543.jpg', '000068539.jpg', '000129457.jpg', '000282317.jpg', '000224958.jpg', '000233721.jpg', '000183534.jpg', '000205311.jpg', '000239212.jpg', '000228121.jpg', '000039036.jpg', '000299242.jpg', '000214383.jpg', '000002402.jpg', '000160659.jpg', '000259989.jpg', '000053939.jpg', '000236380.jpg', '000134004.jpg', '000139456.jpg', '000285190.jpg', '000180705.jpg', '000077915.jpg', '000100141.jpg', '000015709.jpg', '000215235.jpg', '000194445.jpg', '000211820.jpg', '000137458.jpg', '000014637.jpg', '000150568.jpg', '000086941.jpg', '000131484.jpg', '000207255.jpg', '000099545.jpg', '000208467.jpg', '000212343.jpg', '000277126.jpg']


# create a testing dir with name 'test' and 'image' and 'label' subdirectories. clean up the dir if it already exists
# Check if the directory exists

os.makedirs('../cropped/problematic_test/test/image', exist_ok=True)
os.makedirs('../cropped/problematic_test/test/label', exist_ok=True)



#copy the images from the images list to the 'image' subdirectory
for image in images_43:
    shutil.copy(f'../cropped_/data65k/data/test/image/{image}', f'../cropped/problematic_test/test/image/{image}')
    # get the image name without the extension
    image_name = image.split('.')[0]
    # copy the label file to the 'label' subdirectory
    shutil.copy(f'../cropped_/data65k/data/test/label/{image_name}.json', f'../cropped/problematic_test/test/label/{image_name}.json')
