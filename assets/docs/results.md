# Results

### Storage Tests

In these tests, we compare our technique against other downsampling techniques to see which one stores data more compact. More information the experiments can be found in the Experiments section.

#### Storage Test 1: Base Case

| Dataset  | Total Size (KB) | Average Size (KB) | Objects larger than binary | Objects smaller than binary | Objects equal to binary |
|----------|-----------------|-------------------|----------------------------|----------------------------|-------------------------|
| Slice64  |    393952KB     |        32KB       |           N/A              |             N/A            |           N/A           |
| Voxel64  |    272184KB     |        22KB       |          2398              |            9913            |            0            |
| Slice128 |   3151616KB     |       256KB       |           N/A              |             N/A            |           N/A           |
| Voxel128 |    395505KB     |        32KB       |           151              |           12160            |            0            |

In our Base Case study, we compare our Binary Encoding technique against Open3D's Voxelization Technique + a Density-Aware Downsampling method. The Density-Aware Downsampling method was only used if the total number of points in the Voxelized point cloud was higher than the total number of points the Binary Encoded point cloud, in order to keep things fair.

In this test, we find that Voxelization rather handily beats our method at 64 slices, and overwhelmingly beats our method at 128 slices. I initially thought we would perform much worse than voxelization at this point, but was surprised to see just how much worse this method was compared to voxelization.

Although we store 8 points for the price of 1 byte, we also store every single empty space as a point in our bitarray. Voxelization stores every active point more expensively than we do, but does not store the empty space. Our base method can be expected to beat Voxelization when our method is used with lower resolution on large point clouds.

Furthermore, although this is the storage test, it is worth noting that Voxelization performs significantly faster than our Binary Encoding technique. The higher the slice resolution, the longer it takes for a point cloud to be encoded. Although we have significantly decreased the length of time it takes for the binary encoding to finish, the 128 resolution still takes on average just under three seconds to finish, whereas the voxelization takes less than a tenth of a second. However, we find that this length of time is not necessarily a deterrent from using our method in cases where it makes sense to use, as this is a pre-processing step, and the decoding time is exponentially faster than the encoding time.


#### Storage Test 2: RLE Test 1

| Dataset  | Total Size (KB) | Average Size (KB) | Objects larger than binary | Objects smaller than binary | Objects equal to binary |
|----------|-----------------|-------------------|----------------------------|----------------------------|-------------------------|
| Slice64  |     53058KB     |       4.3KB       |           N/A              |             N/A            |           N/A           |
| Voxel64  |    272184KB     |        22KB       |          12081             |             229            |            1            |
| Slice128 |    139476KB     |      11.3KB       |           N/A              |             N/A            |           N/A           |
| Voxel128 |    395505KB     |        32KB       |          11631             |             679            |            1            |

In our Run Length Encoding (RLE) study, we compare our Binary Encoding technique against Open3D's Voxelization Technique + a Density-Aware Downsampling method. The Density-Aware Downsampling method was only used if the total number of points in the Voxelized point cloud was higher than the total number of points the Binary Encoded point cloud, in order to keep things fair. It is important to mention that we did not re-run the Voxelization test for this study, as there was no change to the encoding process that made it necessary to re-run the Voxelization datasets.

In short, our RLE method further compresses the bitarray by representing the length of a sequence as a binary number. Until further literature review is finished, I am calling our method of RLE "Dynamic Binary RLE." Dynamic, because the bit-string length is not a static number for each object being stored, and Binary because it's all 1s and 0s. I go into further detail in the corresponding Experiments section.

In this test, we find that our method beats the Voxelization method in a vast majority of cases. As of right now, it seems the only times Voxelization stores more compact is when point clouds are so small they hardly need to be down-sampled to begin with (such as point clouds with only 100 points or less). We will need to sample more of the cases where Voxelization wins to make this a blanket statement.

Our method finds exponential gains in compression when point clouds are large. This is because we can better utilize the method of storing 8 points in 1 byte of memory than previously, as we have significantly reduced the amount of storage space the empty space of the point cloud takes. In that same vein, we still find sufficiently performed compression when we compare the small and medium sized point clouds.

I will once again state that Voxelization performs significantly faster than our Binary Encoding technique. However, adding this extra encoding step does dramatically further increase the time it takes to perform our technique. On average, for Slice64, we only added an additional 0.01-0.05 seconds, and for Slice128, we only added an additional 0.05-0.1 seconds. As for decoding speeds, the Binary Encoding technique still decodes extremely fast (I am assuming less than 0.1 seconds, but I have not done a rigourous speed test on the decoding aspect yet).

Our implementation of RLE seems to be what we're looking for to enable our Binary Encoding Technique to be an extremely efficient method of compressing point cloud data. Our future tests will involve analyzing the structural integrity of the point cloud post-downsampling, and performing AI/ML tasks on the data.

#### Storage Test 3: RLE Test 2

In round 2 of testing, we re-run both the binary point cloud encoding portion and the voxelization portion, as we have changed the style of the heading and the method of encoding. The change to the encoding reflects a fix to how points were being placed and measured, and the change to the heading reflects the need to keep only necessary information in the stored binary file.

| Dataset  | Total Size (KB) | Average Size (KB) | Objects larger than binary | Objects smaller than binary | Objects equal to binary |
|----------|-----------------|-------------------|----------------------------|----------------------------|-------------------------|
| Slice64  |     56602KB     |       4.6KB       |           N/A              |             N/A            |           N/A           |
| Voxel64  |    280371KB     |        22KB       |          12003             |             308            |            0            |
| Slice128 |    144555KB     |      11.7KB       |           N/A              |             N/A            |           N/A           |
| Voxel128 |    400118KB     |        32KB       |          11481             |             828            |            2            |

Even with this change, our technique is still, on average, superior. In this situation where we have added a substantial number of points per point cloud, there is not a proportial substantial increase in point clouds where voxelization is better than our technique.

## Similarity Tests

### Modified Hausdorff Distance

Below is the results of the Modified Hasudroff Distance test. The first table shows the overall average across all 40 categories for each data set, and in the next four tables are the averages for each individual category within the data set. The test was conducted by comparing the point cloud of the data set with the point cloud of the ground truth data set. Modified Hausdorff Distance compares the nearest point from point cloud A to the nearest point in the same position in point cloud B, then from B to A, and takes the average of these distances for each point cloud.

| Dataset  |    Overall Average    |
|----------|-----------------------|
| Slice64  | 0.00616841119289346   |
| Voxel64  | 0.0017316045592079908 |
| Slice128 | 0.003651373099715794  |
| Voxel128 | 0.001071547890892029  |

Slice64 Dataset:
| Category | Average |
|----------|---------|
| glass_box    | 0.005963584946819966  |
| cone         | 0.008161939538837136  |
| door         | 0.004921292608648928  |
| curtain      | 0.0046358827653057546 |
| plant        | 0.007089397020473635  |
| wardrobe     | 0.0055764583662436746 |
| range_hood   | 0.005969934554295456  |
| dresser      | 0.006067026012599821  | 
| night_stand  | 0.008020660178547113  |
| bookshelf    | 0.005971343505698382  |
| tent         | 0.007407829558777578  | 
| stairs       | 0.0062887925164025965 |
| lamp         | 0.005953704130124918  |
| piano        | 0.0063602716778546654 |
| sink         | 0.005675488250869795  | 
| bathtub      | 0.0054811182499369315 |
| person       | 0.005228046585959122  | 
| xbox         | 0.005384608855458796  | 
| desk         | 0.0065653753127059376 | 
| bench        | 0.005032659312979108  |
| sofa         | 0.00525950720911797   |
| guitar       | 0.004769654090645542  |
| airplane     | 0.005363441176560282  |
| mantel       | 0.00543537314534497   |
| chair        | 0.006133687932438406  |
| laptop       | 0.00686023888231787   |
| cup          | 0.007831868732910884  |
| flower_pot   | 0.008102166747787373  |
| monitor      | 0.0064690409830030745 |
| radio        | 0.005588667414514633  |
| bed          | 0.005964580456454218  |
| toilet       | 0.007169379808854618  |
| bowl         | 0.009507622808837648  |
| stool        | 0.006005565890609723  |
| vase         | 0.0070805965902065    | 
| bottle       | 0.005034806159688385  |
| car          | 0.005077885639131224  | 
| table        | 0.007058947427826375  |
| tv_stand     | 0.0053769013140669624 |
| keyboard     | 0.004891101356882418  |
| **Overall**  | **0.00616841119289346** |

Slice128 Dataset:
| Category     | Average  |
|--------------|----------|
| glass_box    | 0.0039478296993626745 |
| cone         | 0.005656600349856429  |
| door         | 0.0026243515658475733 |
| curtain      | 0.0024855416726011197 |
| plant        | 0.004598279329214552  |
| wardrobe     | 0.0031261762623331366 |
| range_hood   | 0.003338025877092936  |
| dresser      | 0.0035306904035133945 |
| night_stand  | 0.0047275969454734935 |
| bookshelf    | 0.0035930671670213023 |
| tent         | 0.004440278210147685  |
| stairs       | 0.0035972662990093824 |
| lamp         | 0.0030677271414273166 |
| piano        | 0.0035958229748472704 |
| sink         | 0.0031320212655354825 |
| bathtub      | 0.0029164134531141617 |
| person       | 0.002751161845912958  |
| xbox         | 0.002879936099584232  |
| desk         | 0.003513914491934791  |
| bench        | 0.0028238262135414816 |
| sofa         | 0.002942759334817103  |
| guitar       | 0.002431280911392381  |
| airplane     | 0.002829280578050118  |
| mantel       | 0.002993203033442893  |
| chair        | 0.003535765311369045  |
| laptop       | 0.005339254813250016  |
| cup          | 0.005335745901027218  |
| flower_pot   | 0.005292945228143379  |
| monitor      | 0.0041130024047520515 |
| radio        | 0.0029941084277776897 |
| bed          | 0.0032831515315073054 |
| toilet       | 0.004582696237295555  |
| bowl         | 0.006532295671948076  |
| stool        | 0.00424023946474007   |
| vase         | 0.004621626481428855  |
| bottle       | 0.0026823168597664595 |
| car          | 0.002621655502802182  |
| table        | 0.00378623979058814   |
| tv_stand     | 0.0030354649825840227 |
| keyboard     | 0.0025153642545778115 |
| **Overall**  | **0.003651373099715794** |

Voxel64 Dataset:
| Category     | Average  |
|--------------|----------|
| glass_box    | 0.002135709814700037   |
| cone         | 0.007126933620857678   |
| door         | 0.0002580383108318774  |
| curtain      | 0.00014208236230369591 |
| plant        | 0.003128911705716093   |
| wardrobe     | 0.0012044663816342526  |
| range_hood   | 0.0024852895299637503  |
| dresser      | 0.0025509656131737874  |
| night_stand  | 0.005957659474574874   |
| bookshelf    | 0.00036426065364133744 |
| tent         | 0.0018293087961572113  |
| stairs       | 0.0017043276226119256  |
| lamp         | 0.0013583659732334884  |
| piano        | 0.002209900491823238   |
| sink         | 0.0019137399580836055  |
| bathtub      | 0.0006821110690741245  |
| person       | 0.00029613997575113327 |
| xbox         | 0.0009888365093952446  |
| desk         | 0.0018893056461795478  |
| bench        | 0.0003073677419091     |
| sofa         | 0.0004038145798952332  |
| guitar       | 0.00011960584716680565 |
| airplane     | 0.0011408352428393987  |
| mantel       | 0.0021130387365383588  |
| chair        | 0.0019378918185491934  |
| laptop       | 0.0032988913380403813  |
| cup          | 0.0032351456486178288  |
| flower_pot   | 0.0034798007467385186  |
| monitor      | 0.0021398523506169754  |
| radio        | 0.0010698362176765979  |
| bed          | 0.001224271463965189   |
| toilet       | 0.002297856400260394   |
| bowl         | 0.00115423505423575    |
| stool        | 0.0011733333934284115  |
| vase         | 0.002084265449282531   |
| bottle       | 0.00009345650198828729 |
| car          | 0.0003265614589337126  |
| table        | 0.002426364049429799   |
| tv_stand     | 0.000919786767219598   |
| keyboard     | 0.0000916180512806738  |
| **Overall**  | **0.0017316045592079908** |

Voxel128 Dataset:
| Category     | Average  |
|--------------|----------|
| glass_box    | 0.002011316231071899   |
| cone         | 0.006563443011327107   |
| door         | 0.000128963900574608   |
| curtain      | 0.0000691860135211133  |
| plant        | 0.002344675457431967   |
| wardrobe     | 0.00048822023549949305 |
| range_hood   | 0.0014807039057355075  |
| dresser      | 0.0012809002521748177  |
| night_stand  | 0.003437507325560067   |
| bookshelf    | 0.0002472929285574145  |
| tent         | 0.0010022393672773334  |
| stairs       | 0.000966326507583397   |
| lamp         | 0.0004824768061912626  |
| piano        | 0.0011052260134755655  |
| sink         | 0.0008712171321195517  |
| bathtub      | 0.0002154181276352832  |
| person       | 0.00010283588576525989 |
| xbox         | 0.0002596601773798877  |
| desk         | 0.000806579197254563   |
| bench        | 0.00014331504475516196 |
| sofa         | 0.00012544682856702936 |
| guitar       | 0.00010054738793805906 |
| airplane     | 0.0003997642162545134  |
| mantel       | 0.000968162437701336   |
| chair        | 0.0009530678030397227  |
| laptop       | 0.0028460466542872785  |
| cup          | 0.0023531440967105677  |
| flower_pot   | 0.0028173933035647728  |
| monitor      | 0.0012938314240292371  |
| radio        | 0.0003656035627793393  |
| bed          | 0.0005044925099182972  |
| toilet       | 0.0014341683537181012  |
| bowl         | 0.0008380335475826171  |
| stool        | 0.0004942117965793118  |
| vase         | 0.0016684068788628108  |
| bottle       | 0.00003402711831233006 |
| car          | 0.00005719791277688253 |
| table        | 0.001170583440478567   |
| tv_stand     | 0.0003562896397745336  |
| keyboard     | 0.00007399320191459119 |
| **Overall**  | **0.001071547890892029** |

### Chamfer's Distance

Below is the results of the Chamfer's Distance test. The first table shows the overall average across all 40 categories for each data set, and in the next four tables are the averages for each individual category within the data set. The test was conducted by comparing the point cloud of the data set with the point cloud of the ground truth data set. Chamfer's Distance compares the nearest point from point cloud A to the nearest point in the same position in point cloud B, then from B to A, and takes the sum of the minimum distances for each point cloud, and averages it out between the two.


| Dataset  |    Overall Average    |
|----------|-----------------------|
| Slice64  | 0.006782016374157892  |
| Voxel64  | 0.0029915333896178603 |
| Slice128 | 0.004195367120516455  |
| Voxel128 | 0.0017880446132857424 |


Slice64 Dataset:
| Object        |         Average         |
|---------------|-------------------------|
| glass_box     | 0.006482610097117446     |
| cone          | 0.01047009220799138      |
| door          | 0.005014270182225524     |
| curtain       | 0.004800956120406657     |
| plant         | 0.0080327857556276       |
| wardrobe      | 0.005882774930268028     |
| range_hood    | 0.006561542540718477     |
| dresser       | 0.006641800140253564     |
| night_stand   | 0.010104883794478315     |
| bookshelf     | 0.006432575946987018     |
| tent          | 0.008801382390396862     |
| stairs        | 0.0069194617673266874    |
| lamp          | 0.006478379096254504     |
| piano         | 0.007047606709179776     |
| sink          | 0.0060141252088436485    |
| bathtub       | 0.005543830259742346     |
| person        | 0.005262936304427622     |
| xbox          | 0.005608234971206436     |
| desk          | 0.007477997236609773     |
| bench         | 0.00509366896202452      |
| sofa          | 0.005275308007639118     |
| guitar        | 0.004757515569888694     |
| airplane      | 0.005675992661032855     |
| mantel        | 0.005725321325670478     |
| chair         | 0.006589437930667942     |
| laptop        | 0.00802734990747436      |
| cup           | 0.008929597175682343     |
| flower_pot    | 0.00980690067185205      |
| monitor       | 0.007156632180374942     |
| radio         | 0.005827127857187279     |
| bed           | 0.006298414326434479     |
| toilet        | 0.008101409489019622     |
| bowl          | 0.010842438792826151     |
| stool         | 0.006417933528003229     |
| vase          | 0.007894532289138436     |
| bottle        | 0.005142057856600174     |
| car           | 0.004974042750830451     |
| table         | 0.008504526326217678     |
| tv_stand      | 0.0056362506360430565    |
| keyboard      | 0.005025951061646187     |
| **Overall Average** | **0.006782016374157892** |

Slice128 Dataset:
| Object        |         Average         |
|---------------|-------------------------|
| glass_box     | 0.005098019413645553     |
| cone          | 0.00778216844489029      |
| door          | 0.0026306493852658024    |
| curtain       | 0.0025833718748154186    |
| plant         | 0.005566760851193358     |
| wardrobe      | 0.0032410583125007364    |
| range_hood    | 0.0038428464895078366    |
| dresser       | 0.00406370054619951      |
| night_stand   | 0.0061138329473687135    |
| bookshelf     | 0.004090648864009542     |
| tent          | 0.00524335668354026      |
| stairs        | 0.004024258208038621     |
| lamp          | 0.0033176747827665617    |
| piano         | 0.004130518485408845     |
| sink          | 0.003381258439432835     |
| bathtub       | 0.0029948787657279527    |
| person        | 0.002811790315500742     |
| xbox          | 0.0029945089774902718    |
| desk          | 0.004042212959800346     |
| bench         | 0.002917765148635204     |
| sofa          | 0.003042202256492361     |
| guitar        | 0.0024797729610989703    |
| airplane      | 0.0029522331014313077    |
| mantel        | 0.003244324992180118     |
| chair         | 0.003938017855818889     |
| laptop        | 0.00706090289501921      |
| cup           | 0.006371747160941821     |
| flower_pot    | 0.006771681631773284     |
| monitor       | 0.00489538437391969      |
| radio         | 0.003132855543202054     |
| bed           | 0.0035214838488062967    |
| toilet        | 0.005511791284642555     |
| bowl          | 0.007794921641387016     |
| stool         | 0.005041078539392776     |
| vase          | 0.0054295672730598635    |
| bottle        | 0.002759535396525997     |
| car           | 0.0025841224471611947    |
| table         | 0.004517777419851614     |
| tv_stand      | 0.003279306548141366     |
| keyboard      | 0.00261469775407335      |
| **Overall Average** | **0.004195367120516455** |

Voxel64 Dataset:
| Object        |         Average         |
|---------------|-------------------------|
| glass_box     | 0.0036014969781576556    |
| cone          | 0.012738019064063003     |
| door          | 0.00046127018862232827   |
| curtain       | 0.00023763281202057848   |
| plant         | 0.00533911518129075      |
| wardrobe      | 0.0020398311750788696    |
| range_hood    | 0.004563852704846183     |
| dresser       | 0.004482225605817654     |
| night_stand   | 0.010537212642050988     |
| bookshelf     | 0.0006436080986878008    |
| tent          | 0.0032193356482614474    |
| stairs        | 0.003036870617741186     |
| lamp          | 0.0023261417664441997    |
| piano         | 0.003983952990845072     |
| sink          | 0.0033492562620963244    |
| bathtub       | 0.001162350683970318     |
| person        | 0.0005135481086214378    |
| xbox          | 0.0017492546406196092    |
| desk          | 0.0033327752483919195    |
| bench         | 0.0005178158507739297    |
| sofa          | 0.0006648169826202159    |
| guitar        | 0.00017003429866641861   |
| airplane      | 0.002091399393592623     |
| mantel        | 0.0040124126716694335    |
| chair         | 0.0032971540495269356    |
| laptop        | 0.005864241316780886     |
| cup           | 0.004933568065466368     |
| flower_pot    | 0.005761639511878476     |
| monitor       | 0.003701401869222145     |
| radio         | 0.001857774496058681     |
| bed           | 0.002106786230055147     |
| toilet        | 0.0037396631646270414    |
| bowl          | 0.0018339287683071313    |
| stool         | 0.0018993978093606065    |
| vase          | 0.003172059882919926     |
| bottle        | 0.00015198390352039106   |
| car           | 0.0005496325834521708    |
| table         | 0.004267685875200518     |
| tv_stand      | 0.0016188178989041506    |
| keyboard      | 0.00013137054448390548   |
| **Overall Average** | **0.0029915333896178603** |

Voxel128 Dataset:
| Object        |         Average         |
|---------------|-------------------------|
| glass_box     | 0.0036793835546583536    |
| cone          | 0.013051209747374058     |
| door          | 0.0007020124988658595    |
| curtain       | 0.00023496069483964646   |
| plant         | 0.005306501057952643     |
| wardrobe      | 0.0019287864011176825    |
| range_hood    | 0.004476132366296768     |
| dresser       | 0.004551020227789879     |
| night_stand   | 0.010741859797525477     |
| bookshelf     | 0.0007036291442894076    |
| tent          | 0.0032678915569782255    |
| stairs        | 0.0031277013093369       |
| lamp          | 0.002482264611981011     |
| piano         | 0.0040110410639274715    |
| sink          | 0.003429769248537719     |
| bathtub       | 0.0011432640879561684    |
| person        | 0.0005613548086555917    |
| xbox          | 0.0017649522350269268    |
| desk          | 0.0034347848396159826    |
| bench         | 0.000544453317029571     |
| sofa          | 0.0006634793034521458    |
| guitar        | 0.00017823716016685032   |
| airplane      | 0.002117671213978767     |
| mantel        | 0.00407167764943868      |
| chair         | 0.003397491939731668     |
| laptop        | 0.00607719103189695      |
| cup           | 0.005126936153049469     |
| flower_pot    | 0.005894982487007379     |
| monitor       | 0.003785788452252924     |
| radio         | 0.0019245791948347025    |
| bed           | 0.0022430075435042387    |
| toilet        | 0.003845197073028444     |
| bowl          | 0.0019227847640858297    |
| stool         | 0.0019256505582332616    |
| vase          | 0.003256835809796076     |
| bottle        | 0.00016661117610794378   |
| car           | 0.0005700722415828205    |
| table         | 0.004371239859986214     |
| tv_stand      | 0.0016947482270309928    |
| keyboard      | 0.00013672525131461525   |
| **Overall Average** | **0.0031065911914068424** |
