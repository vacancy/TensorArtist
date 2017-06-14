Pix2Pix datasets download toolkits
----------------------------------

## Original dataset downloading

    $ bash ./data/download_dataset.sh dataset_name

Available datasets are:

- `facescrub`: 106k images from [FaceScrub dataset](http://vintage.winklerbros.net/facescrub.html)
- `facades`: 400 images from [CMP Facades dataset](http://cmp.felk.cvut.cz/~tylecr1/facade/).
- `cityscapes`: 2975 images from the [Cityscapes training set](https://www.cityscapes-dataset.com/).
- `maps`: 1096 training images scraped from Google Maps
- `edges2shoes`: 50k training images from [UT Zappos50K
  dataset](http://vision.cs.utexas.edu/projects/finegrained/utzap50k/).
- `edges2handbags`: 137K Amazon Handbag images from [iGAN project](https://github.com/junyanz/iGAN).

## Convert (crop and resize, lmdb construction)

For `edges2shoes` and `edges2handbags` (where AB-pairs are stored in the same image), try:

    $ tart gen-lmdb-pair.py -i DIR_TO_IMGS -o DIR_TO_LMDB/train_edges_db DIR_TO_LMDB/train_shoes_db

For other datasets (e.g. facescrub), try:

    $ tart gen-lmdb-single.py -i DIR_TO_ACTOR_IMGS -o DIR_TO_LMDB/train_actors_db
    $ tart gen-lmdb-single.py -i DIR_TO_ACTRESS_IMGS -o DIR_TO_LMDB/train_actresses_db
