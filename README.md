# Amazon Rekognition Custom Labels Demo

This repository contains an example of using Rekognition custom labels for Object detection. The initial dataset was copied from [Context-based information generation from construction site images using unmanned aerial vehicle (UAV)-acquired data and image captioning](https://data.mendeley.com/datasets/4h68fmktwh/1). The dataset is licensed as [CC BY 4.0](http://creativecommons.org/licenses/by/4.0).

The dataset contains a set of 1431 annotated images in jpg format, named following the regular expression `[1-9][0-9]*.jpg`.

Annotations consist of a collection of `regions` numbered from 0, defined by bounding boxes with a textual description of what was identified, generally o
- Bounding box: each region contains a `shape_attributes` field, containing the following subfields:
  - `name`: usually just "rect", the shape of the box;
  - `x`: region start x reference, as pixes from left (to be confirmed).
  - `y`: region start y reference, as pixes from top (to be confirmed).
  - `width`: width of the region in pixels.
  - `height`: height of the region in pixels.
- Textual description: each region contains a `region_attributes` field, itself containing a `phrase` subfield f the format `<qualifier>* <object> <locative verb/preposition> <location>`:
   - `qualifier` can be a color, a collection, a quantity, a temporal qualifier or others. Many qualifiers can exist.
   - `object` is the class of interest, what is to be identified.
   - `locative verb/preposition` determines the spacial relationship betwen the `object` and the `location`.
   - `location` is a reference area, such as "the ground", "concrete", "structure". It can have its own qualifiers.
