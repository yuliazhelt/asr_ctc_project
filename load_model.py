import gdown

# a file
id = "1XEgdLxCrYNbAv8YC5sTIywWvM52v3tay"
output = "model_best.pth"
gdown.download(id=id, output=output, quiet=False)
