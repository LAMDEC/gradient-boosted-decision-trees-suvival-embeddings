#ifndef XGBOOST_HPP
#define XGBOOST_HPP

#include <iostream>
#include <map>
#include <ostream>
#include <string>
#include <vector>
#include <xgboost/c_api.h>

#define safe_xgboost(call)                                                     \
  {                                                                            \
    int err = (call);                                                          \
    if (err != 0) {                                                            \
      throw std::runtime_error(std::string(__FILE__) + ":" +                   \
                               std::to_string(__LINE__) + ": error in " +      \
                               #call + ":" + XGBGetLastError());               \
    }                                                                          \
  }

int xgboost_embedding() {
  DMatrixHandle matrix_h[1]; // handle to DMatrix
  char config[] = "{\"uri\": \"data.csv?format=csv\", \"silent\": 0}";
  // Load the data from file & store it in data variable of DMatrixHandle
  // datatype
  safe_xgboost(XGDMatrixCreateFromURI(config, &matrix_h[0]));

  char const *feat_names[]{
      "numCDA",
      "id_natureza_divida",
      "agrupamento_situacao",
      "cod_fase_cobranca",
      "cod_situacao_cda",
      "ano_cadastramento",
      "qtde_execucoes_fiscais",
      "ordem_inscricao_cda_imovel",
      "qtde_cdas_pagas_antes_situacao_imovel",
      "qtde_cdas_canceladas_antes_situacao_imovel",
      "qtde_cdas_ativas_antes_situacao_imovel",
      "flag_historico_leilao",
      "flag_historico_parcelada",
      "flag_historico_suspensa",
      "flag_historico_arrematacao",
      "flag_historico_negociada",
      "flag_historico_parcelamento_irregular",
      "flag_ultima_situacao_ajuizamento",
      "qtde_max_parcelas_todas_guias",
      "qtde_soma_parcelas_pagas_todas_guias",
      "qtde_parcelas_ultima_guia",
      "qtde_parcelas_pagas_ultima_guia",
      "qtde_guias_avista_emitidas",
      "qtde_guias_parceladas_emitidas",
      "qtde_guias_regularizacao_emitidas",
      "valor_saldo_atualizado",
      "qtde_cdas_pagas_antes_situacao_pessoa",
      "qtde_cdas_canceladas_antes_situacao_pessoa",
      "qtde_cdas_ativas_antes_situacao_pessoa",
      "valor_cdas_pagas_antes_situacao_pessoa",
      "valor_cdas_canceladas_antes_situacao_pessoa",
      "valor_cdas_ativas_antes_situacao_pessoa",
      "cod_tipo_pessoa_dam",
      "qtde_meses_esquecimento_cda",
      "ano_ultima_movimentacao",
      "qtde_mudancas_estado",
  };
  safe_xgboost(
      XGDMatrixSetStrFeatureInfo(matrix_h[0], "feature_name", feat_names, 36));

  // i for integer, q for quantitive, c for categorical.  Similarly "int" and
  // "float" are also recognized.
  char const *feat_types[]{"i", "i", "i", "i", "i", "i", "i", "i", "i",
                           "i", "i", "i", "i", "i", "i", "i", "i", "i",
                           "i", "q", "i", "i", "i", "i", "i", "q", "i",
                           "i", "i", "q", "q", "q", "i", "i", "i", "i"};
  safe_xgboost(
      XGDMatrixSetStrFeatureInfo(matrix_h[0], "feature_type", feat_types, 36));

  char const **c_out_features = NULL;
  bst_ulong out_size = 36;

  // Asumming the feature names are already set by `XGDMatrixSetStrFeatureInfo`.
  safe_xgboost(XGDMatrixGetStrFeatureInfo(matrix_h[0], "feature_name",
                                          &out_size, &c_out_features));

  for (bst_ulong i = 0; i < out_size; ++i) {
    // Here we are simply printing the string.  Copy it out if the feature name
    // is useful after printing.
    printf("feature %lu: %s\n", i, c_out_features[i]);
  }

  float labels_lower[100];
  float labels_upper[100];
  for (int i = 0; i < 100; i++) {
    labels_lower[i] = i + i * i * i;
    labels_upper[i] = i + i * i * i;
  }
  safe_xgboost(XGDMatrixSetFloatInfo(matrix_h[0], "label_lower_bound",
                                     labels_lower, 100));
  safe_xgboost(XGDMatrixSetFloatInfo(matrix_h[0], "label_upper_bound",
                                     labels_upper, 100));

  BoosterHandle booster_h;
  safe_xgboost(XGBoosterCreate(matrix_h, 1, &booster_h));

  safe_xgboost(XGBoosterSetParam(booster_h, "verbosity", "3"));
  safe_xgboost(XGBoosterSetParam(booster_h, "device", "cpu"));
  safe_xgboost(XGBoosterSetParam(booster_h, "booster", "dart"));
  safe_xgboost(XGBoosterSetParam(booster_h, "objective", "survival:aft"));
  safe_xgboost(XGBoosterSetParam(booster_h, "aft_loss_distribution", "normal"));

  for (int i = 0; i < 6; i++) {
    safe_xgboost(XGBoosterUpdateOneIter(booster_h, i, matrix_h[0]))
  }

  bst_ulong num_trees;
  const char **dump;

  safe_xgboost(XGBoosterDumpModel(booster_h, "", 1, &num_trees, &dump));

  for (size_t i = 0; i < num_trees; i++) {
    std::cout << dump[i] << std::endl;
  }

  std::vector<std::vector<int>> embedding(100);

  for (size_t i = 0; i < num_trees; i++) {
    BoosterHandle tree;
    safe_xgboost(XGBoosterSlice(booster_h, i, i + 1, 1, &tree));

    const bst_ulong *out_shape;
    bst_ulong out_dim;
    const float *out_result;
    char config_predict[] =
        "{\"type\": 6, \"training\": false, \"iteration_begin\": "
        "0, \"iteration_end\": 0, \"strict_shape\": false}";

    safe_xgboost(XGBoosterPredictFromDMatrix(
        tree, matrix_h[0], config_predict, &out_shape, &out_dim, &out_result));

    std::cout << "\nÁrvore " << i << "\n--------------------"
              << "\nout_dim: " << out_dim << "\nn_samples: " << out_shape[0]
              << std::endl;

    std::map<int, int> map;
    std::cout << "\nPrevisões da árvore " << i << ": ";
    for (size_t j = 0; j < out_shape[0]; j++) {
      std::cout << out_result[j] << " ";
      map.emplace(out_result[j], map.size());
    }
    std::cout << std::endl;
    for (const auto &[k, v] : map) {
      std::cout << "m[" << k << "]" << " = " << v << std::endl;
    }

    for (size_t j = 0; j < out_shape[0]; j++) {
      std::vector<int> treeEmbedding(map.size(), 0);
      treeEmbedding[map[out_result[j]]] = 1;
      embedding[j].insert(embedding[j].end(), treeEmbedding.begin(),
                          treeEmbedding.end());
    }
  }

  for (int i = 0; i < 100; i++) {
    for (const auto n : embedding[i]) {
      std::cout << n << " ";
    }
    std::cout << std::endl;
  }

  /*safe_xgboost(XGBoosterPredictFromDMatrix(booster_h, matrix_h[0],*/
  /*                                         config_predict, &out_shape,
   * &out_dim,*/
  /*                                         &out_result));*/

  /*std::cout << "\nn_samples: " << out_shape[0] << "\nn_iterations: "*/
  /*          << out_shape[1]*/
  /*          << "\nout_dim: " << out_dim << std::endl;*/

  /*std::cout << "\nPRED:\n";*/
  /*for (size_t i = 0; i < out_shape[0] * out_shape[1]; i++) {*/
  /*  std::cout << out_result[i] << " ";*/
  /*}*/
  /*std::cout << std::endl;*/

  safe_xgboost(XGDMatrixFree(matrix_h[0]));
  safe_xgboost(XGBoosterFree(booster_h));

  std::cout << "xgboost" << std::endl;

  return 0;
}

#endif
