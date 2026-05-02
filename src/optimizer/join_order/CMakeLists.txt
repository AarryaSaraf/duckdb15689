add_library_unity(
  duckdb_optimizer_join_order
  OBJECT
  query_graph.cpp
  join_relation_set.cpp
  join_node.cpp
  join_order_optimizer.cpp
  cardinality_estimator.cpp
  cost_model.cpp
  plan_enumerator.cpp
  relation_manager.cpp
  query_graph_manager.cpp
  relation_statistics_helper.cpp
)

if(BUILD_FEEDBACK_LEGACY_SHELL)
  add_library_unity(
    duckdb_optimizer_join_order_feedback
    OBJECT
    legacy_feedback/query_graph.cpp
    legacy_feedback/join_relation_set.cpp
    legacy_feedback/join_node.cpp
    legacy_feedback/join_order_optimizer.cpp
    legacy_feedback/cardinality_estimator.cpp
    legacy_feedback/cost_model.cpp
    legacy_feedback/plan_enumerator.cpp
    legacy_feedback/relation_manager.cpp
    legacy_feedback/query_graph_manager.cpp
    legacy_feedback/relation_statistics_helper.cpp
  )
  target_include_directories(duckdb_optimizer_join_order_feedback PRIVATE
                              "${CMAKE_SOURCE_DIR}/third_party")
endif()

# Vendored nlohmann/json single header (used by NDJSON debug blocks in cardinality_estimator.cpp).
target_include_directories(duckdb_optimizer_join_order PRIVATE
                            "${CMAKE_SOURCE_DIR}/third_party")

set(ALL_OBJECT_FILES
    ${ALL_OBJECT_FILES} $<TARGET_OBJECTS:duckdb_optimizer_join_order>
    PARENT_SCOPE)