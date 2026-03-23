Notes from diffrent sources for each section 

### introduction 
- ## Introduction 
## v3
### 1.1 Overview of resolution standards
 - dirven moore's law - whichpredicts the foubling of transitors density approx every 2 years - has reduec feature sizs of com ships to  nano scale (https://doi.org/10.1109/ICSICT.2004.1434947)
 [ insert image of tranistors size]

- Making characterisation using optical micrsope or visible light impractical ( visible light wavelneght 380 nm to 700nm)
- necessitating thened for higher precision instruments
    - AFM , SEM, TEM, EUV whihc are systems cabable of charcaterising nano meter or sub nano meter scale
    - howveer hte accuracy of such measurements depends on the callibration quality [https://www.nist.gov/news-events/news/2016/03/improving-cd-afm-measurements-tip-down]
- which is where resolution or calibration standards come in
    - These standards typically consist of periodic patterns such as grids, gratings, or spherical particles with well-characterized dimensions at the nanometer scale, providing functionality for magnification calibration, distortion correction, resolution testing, and astigmatism correction.
    - [ insert image of different kinds of resolution standards]

    - although there are numerous different standards this report will focus on the grid resolution standards, which are typically used in the calibration of AFM , CD AFM , EUV, SEM and more
    - [ insert image of grid resolution]
        - But why are they used instead of tin balls or other resolution standards causing the calibration of SPM calivbrationas ( which contains a subset of afm and  stm ) as an example 
        [N. G. Orji & R. G. Dixson “3D-AFM Measurements for Semiconductor
        Structures and Devices” In Metrology and Diagnostic Techniques for
        Nanoelectronics (eds. Z. Ma, & D. G. Seiler) (Pan Stanford, 2017).]
        [Part IV Calibration – Overview
        Nanoscale Calibration Standards and Methods: Dimensional and Related Measurements in the Micro- and Nanometer Range.
        Edited by Gunter Wilkening, Ludger Koenders Copyright c© 2005 Wiley-VCH Verlag GmbH & Co. KGaA, Weinheim
        ISBN: 3-527-40502-X]
        Both of these ustilize a tip that drags or taps across the subjects surface with extremenly high accuracy 
        during the calibration process for  SPM a resolution grid is used one with known characteristics like step height, line width etc to calibrate the x n y axis to look for regularity in either axis
        while noies in th ez0 axis is typically calibrated my checking teh surface roughness, in this case it would be iedal to have a smooth surface

        - additionally it can be used in electron microscopy techniques (SEM and EUV)
            [https://ravescientific.com/resources/education/33-calibrating-a-scanning-electron-microscope-sem]
            - requires line with well knwon pitch ( distance betwen the lines) for magnification calibration distance
            - [https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.250-96.pdf]
            - !help me write abit more on this
        - that is not to say that tin ball or other resolution stnadrs lack use cases , tin spherea re typically used  for exposure and light testing given their on standard apcaing
        - however given the fersitility of resolutio grids for ,multiple calibration practices  this report and project opt to focus on them, particaulary in the novelty of creating  more prependicular side walls

### the importance of sidewall angles in grid resolution standards
- in cases of CD-afm and electron miscropy (SEM,EUV), the perpendicularty of side wall angles matters alot 
- [ insert diagram of the differnce in side wall angle to output of the EUV - [https://doi.org/10.1016/j.mee.2017.06.007]]
- [ !insert image of the electrons hitting the side wall angle and then sacterring in diffrent directions]
- Of particular importance is the verticality of the sidewall profiles of patterned features: a deviation of just 5° from the ideal 90° sidewall angle has been shown to produce a critical dimension (CD) error of up to 20% in a 16 nm line-space pattern  [https://doi.org/10.1016/j.mee.2017.06.007]

- in the afm calibration [https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=918090]
    - !help me write up something about how side wall perpendicularity  matters

- in escense the perpendicularity of the straight wall is not as important as the accuracy to the angle of the perpendicualrity 

### Deliverbles 
This project aim to create a resolution grid with a smooth surface and a wall perpendiculatiry of ( *some value in need to double check*)





## v2

with the continuous shrinking of computer chips optical microscopes are no longer capable of examing the surface features in nanometers [ find ref for reducing chip size] 

Instead reasearches use tool such as CD-AFM - which measures the features by draging a tip across the surface(Part IV
Calibration – Overview
Nanoscale Calibration Standards and Methods: Dimensional and Related Measurements in the Micro- and Nanometer Range ) or EUV which generates ( explanantion on how Euv materials characterisation works ) [https://onlinelibrary.wiley.com/doi/epdf/10.1002/phvs.201900027]

However the accuracy of measurements depends entirely on the calibration. To address this need various resolution or callibration standards are made

 grid structures used commonly in afm, spm calibration of axis characterizes with step heights  (Part IV
Calibration – Overview
Nano scale Calibration Standards and Methods: Dimensional and Related Measurements in the Micro- and Nanometer Range.)
However this brings upon another issue; as microscopy is push to higher and higher accuracy so does the need to create resolution standards grids with greater accuracy 
current techniques such as CD-AFM which can characterized things like side wall angles
[ insert image of rise in accuracy of characterizing techniques ]
[ insert image of resolution grids change in side wall accuracy ]

another common resolution standard are nano scale tin balls with their varied spacing is used for exposure and light testing. The following project focus on the lithography method of fabricating grid resolution standards to address the technical gap in the fabrication of resolution grids 

- what are resolution standards used for 
    calibration of AFM example
- the different kinds of resolution standards and what are they used for
    - there is a novelty in making more precise grid structures because there is a need for more precise grid structures in  .... 
    - the impact of an ideal grid  

### 1.2 nano-fabrication methods
- lithography 
- nanoimprinting 
- someother method



### 1.3 An ideal grid resolution standard
- what is an ideal grid resolution standard
   - defining side wall of 90 degress with sub 1 degress uncertainty 
   

### weee my notes while writing 
validating the logic

computer chips get smaller -> micrscopy / material charactistic methods get more prescies ( afm, cd-afm,EUV,SEM) -> accuracy depends on callibration  -> how do you makemore accurate callibration ( conclusiszely define the chracteristic of resolution standrad)

- how do we conclusisvely show the uncertainly reducing for both material cahracterristic method and the grid resolution standard
- i also need to prove why i want to do this for grid rather than tin balls or other resolution standrad
    - 