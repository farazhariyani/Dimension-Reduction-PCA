#load library
library(readr)

#load dataset
wine <- read.csv(file.choose())
wine

# Hierarchical Clustering
# Normalize the data
normalized_data <- scale(wine) 
summary(normalized_data)

# Distance matrix
d <- dist(normalized_data, method = "euclidean") 

fit <- hclust(d, method = "complete")

# Display dendrogram
plot(fit) 
plot(fit, hang = -1)

groups <- cutree(fit, k = 3) # Cut tree into 3 clusters
rect.hclust(fit, k = 3, border = "red")

membership <- as.matrix(groups)
final <- data.frame(membership, wine)
aggregate(wine, by = list(final$membership), FUN = mean)

write_csv(final, "PCA-HierarchicalClustering-Assignment.csv")
getwd()

# K means clustering
# Elbow curve to decide the k value
twss <- NULL
for (i in 2:14) {
  twss <- c(twss, kmeans(normalized_data, centers = i)$tot.withinss)
}
twss

# Look for an "elbow" in the scree plot
plot(2:14, twss, type = "b", xlab = "Number of Clusters", ylab = "Within groups sum of squares")
title(sub = "K-Means Clustering Scree-Plot")

# 4 Cluster Solution
fit <- kmeans(normalized_data, 4) 
str(fit)
fit$cluster
final <- data.frame(fit$cluster, wine) # Append cluster membership
aggregate(wine, by = list(fit$cluster), FUN = mean)

write_csv(final, "PCA-K-means-Clustering-Assignment.csv")
getwd()

# PCA
attach(wine)

pcaObj <- princomp(wine, cor = TRUE, scores = TRUE, covmat = NULL)
str(pcaObj)
summary(pcaObj)
loadings(pcaObj)
plot(pcaObj) # graph showing importance of principal components 
biplot(pcaObj)

plot(cumsum(pcaObj$sdev * pcaObj$sdev) * 100 / (sum(pcaObj$sdev * pcaObj$sdev)), type = "b")

pcaObj$scores
pcaObj$scores[, 1:5]

# Top 3 pca scores 
final <- cbind(wine, pcaObj$scores)
View(final)

# Scatter diagram
plot(final$Comp.1, final$Comp.2)