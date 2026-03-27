#pragma once
#include "simulation/registry.hpp"
#include <memory>
#include <vector>

namespace moonai {

class System {
public:
  virtual ~System() = default;
  virtual void update(Registry &registry) = 0;
  virtual const char *name() const = 0;
};

class SystemScheduler {
public:
  void add_system(std::unique_ptr<System> system);
  void update(Registry &registry);

  size_t system_count() const {
    return systems_.size();
  }
  void clear() {
    systems_.clear();
  }

private:
  std::vector<std::unique_ptr<System>> systems_;
};

} // namespace moonai
