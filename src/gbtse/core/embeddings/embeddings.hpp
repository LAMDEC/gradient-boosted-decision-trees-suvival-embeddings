#ifndef EMBEDDINGS_HPP
#define EMBEDDINGS_HPP

#include <string>
#include <vector>

class Embedder {
public:
  virtual ~Embedder() {};

  virtual std::vector<std::vector<int>> getEmbedding() = 0;
  virtual void readCSV(std::string filename) = 0;
};

#endif
