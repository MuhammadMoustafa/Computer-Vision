# Part 2: Written

You answers to this part should be included in the `Part2.md` file in the repository of your assignment.


Also, your answers should be formatted as *Markdown* syntax.

## Q1

Transform a few more (easy) RGB values manually into corresponding HSI values.

### Answer

##### H=Cos^(-1) (((R-G)+ (R-B))/(2√(〖(R-G)〗^2+(R-B)(G-B))))
##### S=1-3 min(R,G,B)/I
##### I=(R+G+B)/3

##### e.g. RGB (100, 200, 130)  =  HSI(137, 0.301, 0.561);

## Q2

In the CIE’s RGB colour space (which models human colour perception), the scalars R, G, or B may also be negative. Provide a physical interpretation (obviously, we cannot subtract light from a given spectrum).

### Answer

RGB primaries are/were used and there are certainly many spectral colors (monochromatic lights) that cannot be matched by positive amounts of RGB. (The number depends on the selection of RGB.) The key here is the word positive. What is done is that the light to be matched is desaturated with one of the RGB primaries and then that desaturated light is matched using the other two. For example, 500nm is a saturated cyanish wavelength that is outside almost any RGB gamut. What is done is that it is mixed with a little bit of R and that mixture is matched by a combination of B and G as follows. 

500nm + R matches G + B 

Grassmann´s laws tell us that additive color matches follow simple algebra (that´s why the whole system works at all) and that allows us to subtract the amount of R from both sides of the match resulting in 

500nm matches G + B - R 

Thus, we can have negative amounts of light to match every possible color. That is why plots of the color matching functions in terms of RGB go negative at some wavelengths.
