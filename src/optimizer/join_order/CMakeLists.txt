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

# --- ADDED FOR JSON SUPPORT ---
# 1. Tell CMake to look for headers in the Homebrew directory (standard for Apple Silicon)
target_include_directories(duckdb_optimizer_join_order SYSTEM PRIVATE /opt/homebrew/include)

# 2. If you want to use the find_package approach for portability:
find_package(nlohmann_json QUIET)
if (nlohmann_json_FOUND)
    target_link_libraries(duckdb_optimizer_join_order PRIVATE nlohmann_json::nlohmann_json)
endif()
# ------------------------------

set(ALL_OBJECT_FILES
    ${ALL_OBJECT_FILES} $<TARGET_OBJECTS:duckdb_optimizer_join_order>
    PARENT_SCOPE)