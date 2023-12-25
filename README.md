# Image clustering

Edit (Dec 2023): [clustimage](https://github.com/erdogant/clustimage) seems to provide a convenient interface around what I was trying to do, below.  Let's try that, for simplicity!

Playing around with various clustering libraries and preprocessing steps to group together similar-looking images.

I originally started this after [Google deprecated](https://9to5google.com/2021/09/28/gboard-minis-going-away-october/) their ["Minis" custom stickers](https://ai.googleblog.com/2017/05/neural-network-generated-illustrations.html). I used [Takeout](https://takeout.google.com) to download my Hangouts/Chat history, but there were multiple copies of each sticker that I used multiple times. Seeing as that I wanted to practice doing clustering, this was the perfect opportunity to put this to use.

| ![Example image](https://github.com/rtkfan/image-clusters/blob/master/example.png) |
| -- |
| Example Sweet Mini. I've sent hundreds of these. Don't judge me! |

The actual clustering is based largely on [this article](https://towardsdatascience.com/how-to-cluster-images-based-on-visual-similarity-cd6e7209fe34). I'll run this with a variety of cluster counts, and see which one gives me the lowest k-means inertia. As separate instances of these images should be effectively identical, I expect the inertia to actually be 0 when I've identified the right number of clusters.

I'll use this to identify individual images, and then upload a deduplicated set somewhere I can continue to send these!
