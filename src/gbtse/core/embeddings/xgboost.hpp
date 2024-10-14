#ifndef XGBOOST_HPP
#define XGBOOST_HPP

#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#include <xgboost/c_api.h>

#include "./embedder.hpp"

void inline safe_xgboost(int code) {
  if (code != 0) {
    throw std::runtime_error(std::string(__FILE__) + ":" +
                             std::to_string(__LINE__) + ": error in" + ": " +
                             XGBGetLastError());
  }
}

class XGBoostEmbedder : virtual Embedder {
public:
  void readCSV(std::string filename) override {
    std::string config =
        "{\"uri\": \"" + filename + "?format=csv\", \"silent\": 0}";

    safe_xgboost(XGDMatrixCreateFromURI(config.c_str(), &matrixHandle[0]));

    // artificial labels
    // TODO: get actual labels
    float labels_lower[100];
    float labels_upper[100];
    for (int i = 0; i < 100; i++) {
      labels_lower[i] = i + i * i * i;
      if (i > 50)
        labels_upper[i] = std::numeric_limits<float>::infinity();
      else
        labels_upper[i] = i + i * i * i;
    }
    safe_xgboost(XGDMatrixSetFloatInfo(matrixHandle[0], "label_lower_bound",
                                       labels_lower, 100));
    safe_xgboost(XGDMatrixSetFloatInfo(matrixHandle[0], "label_upper_bound",
                                       labels_upper, 100));
  }

  std::vector<std::vector<int>> getEmbedding() override {
    safe_xgboost(XGBoosterCreate(matrixHandle, 1, &boosterHandle));

    int numTrees = 6;
    trainBooster(numTrees);

    return createEmbedding(numTrees);
  }

private:
  DMatrixHandle matrixHandle[1];
  BoosterHandle boosterHandle;

  void trainBooster(int numTrees) {
    safe_xgboost(XGBoosterSetParam(boosterHandle, "verbosity", "3"));
    safe_xgboost(XGBoosterSetParam(boosterHandle, "device", "cpu"));
    safe_xgboost(XGBoosterSetParam(boosterHandle, "booster", "dart"));
    safe_xgboost(XGBoosterSetParam(boosterHandle, "objective", "survival:aft"));
    safe_xgboost(
        XGBoosterSetParam(boosterHandle, "aft_loss_distribution", "normal"));

    for (int i = 0; i < numTrees; i++) {
      safe_xgboost(XGBoosterUpdateOneIter(boosterHandle, i, matrixHandle[0]));
    }
  }

  std::vector<std::vector<int>> createEmbedding(int numTrees) {
    std::vector<std::vector<int>> embedding(100);
    for (int i = 0; i < numTrees; i++) {
      createTreeEmbedding(embedding, i);
    }
    return embedding;
  }

  void createTreeEmbedding(std::vector<std::vector<int>> &embedding, int i) {
    BoosterHandle tree;
    safe_xgboost(XGBoosterSlice(boosterHandle, i, i + 1, 1, &tree));

    const bst_ulong *out_shape;
    bst_ulong out_dim;
    const float *out_result;
    char config_predict[] =
        "{\"type\": 6, \"training\": false, \"iteration_begin\": "
        "0, \"iteration_end\": 0, \"strict_shape\": false}";

    safe_xgboost(XGBoosterPredictFromDMatrix(tree, matrixHandle[0],
                                             config_predict, &out_shape,
                                             &out_dim, &out_result));

    // Tree embedding algorithm given the leafs:
    // out_result[j] will contain the leaf which the j-th input landed on.
    // For each new leaf, the map will assign a unique position for it with
    // emplace(leaf, map_size). Then, each tree embedding will have the size of
    // the map (which is equal to the number of different leafs) with a 1 on the
    // position assigned for it by the map.

    std::map<int, int> map;

    for (size_t j = 0; j < out_shape[0]; j++) {
      map.emplace(out_result[j], map.size());
    }

    for (size_t j = 0; j < out_shape[0]; j++) {
      std::vector<int> treeEmbedding(map.size(), 0);
      treeEmbedding[map[out_result[j]]] = 1;
      embedding[j].insert(embedding[j].end(), treeEmbedding.begin(),
                          treeEmbedding.end());
    }
  }
};

#endif
