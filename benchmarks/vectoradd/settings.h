#ifndef __SETTINGS_H__
#define __SETTINGS_H__

#include <cuda.h>
#include <getopt.h>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "settings.h"

struct Settings {
  uint32_t cudaDevice;
  uint64_t cudaDeviceId;
  const char* blockDevicePath;
  const char* controllerPath;
  uint64_t controllerId;
  uint32_t adapter;
  uint32_t segmentId;
  uint32_t nvmNamespace;
  bool doubleBuffered;
  size_t numReqs;
  size_t numPages;
  size_t startBlock;
  bool stats;
  const char* input;
  const char* output;
  const char* input_a;
  const char* input_b;
  size_t afileoffset;
  size_t bfileoffset;
  size_t type;
  size_t memalloc;
  size_t numThreads;
  uint32_t domain;
  uint32_t bus;
  uint32_t devfn;
  uint32_t n_ctrls;
  size_t blkSize;
  size_t queueDepth;
  size_t numQueues;
  size_t pageSize;
  uint64_t numElems;
  size_t repeat;
  size_t src;
  uint64_t maxPageCacheSize;
  uint64_t stride;
  uint64_t coarse;
  uint64_t n_elems;
  Settings();
  void parseArguments(int argc, char** argv);

  static std::string usageString(const std::string& name);

  std::string getDeviceBDF() const;
};

struct OptionIface;
using std::make_shared;
using std::string;
using std::vector;
typedef std::shared_ptr<OptionIface> OptionPtr;
typedef std::map<int, OptionPtr> OptionMap;

struct OptionIface {
  const char* type;
  const char* name;
  const char* description;
  const char* defaultValue;
  int hasArgument;

  virtual ~OptionIface() = default;

  OptionIface(const char* type, const char* name, const char* description)
      : type(type),
        name(name),
        description(description),
        hasArgument(no_argument) {}

  OptionIface(const char* type, const char* name, const char* description,
              const char* dvalue)
      : type(type),
        name(name),
        description(description),
        defaultValue(dvalue),
        hasArgument(no_argument) {}

  virtual void parseArgument(const char* optstr, const char* optarg) = 0;

  virtual void throwError(const char*, const char* optarg) const {
    throw string("Option ") + name + string(" expects a ") + type +
        string(", but got `") + optarg + string("'");
  }
};

template <typename T>
struct Option : public OptionIface {
  T& value;

  Option() = delete;
  Option(Option&& rhs) = delete;
  Option(const Option& rhs) = delete;

  Option(T& value, const char* type, const char* name, const char* description)
      : OptionIface(type, name, description), value(value) {
    hasArgument = required_argument;
  }

  Option(T& value, const char* type, const char* name, const char* description,
         const char* dvalue)
      : OptionIface(type, name, description, dvalue), value(value) {
    hasArgument = required_argument;
  }

  void parseArgument(const char* optstr, const char* optarg) override;
};

template <>
void Option<uint32_t>::parseArgument(const char* optstr, const char* optarg) {
  char* endptr = nullptr;

  value = std::strtoul(optarg, &endptr, 0);

  if (endptr == nullptr || *endptr != '\0') {
    throwError(optstr, optarg);
  }
}

template <>
void Option<uint64_t>::parseArgument(const char* optstr, const char* optarg) {
  char* endptr = nullptr;

  value = std::strtoul(optarg, &endptr, 0);

  if (endptr == nullptr || *endptr != '\0') {
    throwError(optstr, optarg);
  }
}

template <>
void Option<bool>::parseArgument(const char* optstr, const char* optarg) {
  string str(optarg);
  std::transform(str.begin(), str.end(), str.begin(),
                 std::ptr_fun<int, int>(std::tolower));

  if (str == "false" || str == "0" || str == "no" || str == "n" ||
      str == "off" || str == "disable" || str == "disabled") {
    value = false;
  } else if (str == "true" || str == "1" || str == "yes" || str == "y" ||
             str == "on" || str == "enable" || str == "enabled") {
    value = true;
  } else {
    throwError(optstr, optarg);
  }
}

template <>
void Option<const char*>::parseArgument(const char* optstr,
                                        const char* optarg) {
  if (optarg == nullptr) {
    throwError(optstr, optarg);
  }
  value = optarg;
}

struct Range : public Option<uint64_t> {
  uint64_t lower;
  uint64_t upper;

  Range(uint64_t& value, uint64_t lo, uint64_t hi, const char* name,
        const char* description, const char* dv)
      : Option<uint64_t>(value, "count", name, description, dv),
        lower(lo),
        upper(hi) {}

  void throwError(const char*, const char*) const override {
    if (upper != 0 && lower != 0) {
      throw string("Option ") + name + string(" expects a value between ") +
          std::to_string(lower) + " and " + std::to_string(upper);
    } else if (lower != 0) {
      throw string("Option ") + name + string(" must be at least ") +
          std::to_string(lower);
    }
    throw string("Option ") + name + string(" must lower than ") +
        std::to_string(upper);
  }

  void parseArgument(const char* optstr, const char* optarg) override {
    Option<uint64_t>::parseArgument(optstr, optarg);

    if (lower != 0 && value < lower) {
      throwError(optstr, optarg);
    }

    if (upper != 0 && value > upper) {
      throwError(optstr, optarg);
    }
  }
};

static void setBDF(Settings& settings) {
  cudaDeviceProp props;

  cudaError_t err = cudaGetDeviceProperties(&props, settings.cudaDevice);
  if (err != cudaSuccess) {
    throw string("Failed to get device properties: ") + cudaGetErrorString(err);
  }

  settings.domain = props.pciDomainID;
  settings.bus = props.pciBusID;
  settings.devfn = props.pciDeviceID;
}

string Settings::getDeviceBDF() const {
  using namespace std;
  ostringstream s;

  s << setfill('0') << setw(4) << hex << domain << ":" << setfill('0')
    << setw(2) << hex << bus << ":" << setfill('0') << setw(2) << hex << devfn
    << ".0";

  return s.str();
}

string Settings::usageString(const string& name) {
  // return "Usage: " + name + " --ctrl=identifier [options]\n"
  //+  "   or: " + name + " --block-device=path [options]";
  return "\n";
}

static string helpString(const string& /*name*/, OptionMap& options) {
  using namespace std;
  ostringstream s;

  s << "" << left << setw(16) << "OPTION" << setw(2) << " " << setw(16)
    << "TYPE" << setw(10) << "DEFAULT" << setw(36) << "DESCRIPTION" << endl;

  for (const auto& optPair : options) {
    const auto& opt = optPair.second;
    s << "  " << left << setw(16) << opt->name << setw(16) << opt->type
      << setw(10) << (opt->defaultValue != nullptr ? opt->defaultValue : "")
      << setw(36) << opt->description << endl;
  }

  return s.str();
}

static void createLongOptions(vector<option>& options, string& optionString,
                              const OptionMap& parsers) {
  options.push_back(option{
      .name = "help", .has_arg = no_argument, .flag = nullptr, .val = 'h'});
  optionString = ":h";

  for (const auto& parserPair : parsers) {
    int shortOpt = parserPair.first;
    const OptionPtr& parser = parserPair.second;

    option opt;
    opt.name = parser->name;
    opt.has_arg = parser->hasArgument;
    opt.flag = nullptr;
    opt.val = shortOpt;

    options.push_back(opt);

    if ('0' <= shortOpt && shortOpt <= 'z') {
      optionString += (char)shortOpt;
      if (parser->hasArgument == required_argument) {
        optionString += ":";
      }
    }
  }

  options.push_back(
      option{.name = nullptr, .has_arg = 0, .flag = nullptr, .val = 0});
}

static void verifyCudaDevice(int device) {
  int deviceCount = 0;

  cudaError_t err = cudaGetDeviceCount(&deviceCount);
  if (err != cudaSuccess) {
    throw string("Unexpected error: ") + cudaGetErrorString(err);
  }

  if (device < 0 || device >= deviceCount) {
    throw string("Invalid CUDA device: ") + std::to_string(device);
  }
}

static void verifyNumberOfThreads(size_t numThreads) {
  size_t i = 0;

  while ((1ULL << i) <= 32) {
    if ((1ULL << i) == numThreads) {
      return;
    }

    ++i;
  }

  throw string("Invalid number of threads, must be a power of 2");
}

void Settings::parseArguments(int argc, char** argv) {
  OptionMap parsers = {
      {'a', OptionPtr(new Option<const char*>(
                input_a, "path", "input_a",
                "File path A file is. Provide .bel path."))},
      {'b', OptionPtr(new Option<const char*>(
                input_b, "path", "input_b",
                "File path B file is. Provide .bel path."))},
      {'A',
       OptionPtr(new Range(
           afileoffset, 0, (uint64_t)std::numeric_limits<uint64_t>::max,
           "aoffset",
           "Offset where the input file contents need to be stored in NVMe SSD",
           "0"))},
      {'B',
       OptionPtr(new Range(
           bfileoffset, 0, (uint64_t)std::numeric_limits<uint64_t>::max,
           "boffset",
           "Offset where the input file contents need to be stored in NVMe SSD",
           "0"))},
      {'v',
       OptionPtr(new Range(
           type, 0, 50, "impl_type",
           "BASELINE=0, COALESCE = 1, COALESCE_CHUNK = 2, BASELINE_PC=3, "
           "COALESCE_PC = 4, COALESCE_CHUNK_PC = 5\n BASELINE_HASH = 6, "
           "COALESCE_HASH = 7, BASELINE_HASH_PC = 9, COALESCE_HASH_PC = 10",
           "1"))},
      {'m', OptionPtr(new Range(
                memalloc, 0, 6, "memalloc",
                "GPUMEM = 0, UVM_READONLY = 1, UVM_DIRECT = 2, BAFS_DIRECT = 6",
                "2"))},
      {'s', OptionPtr(new Option<uint64_t>(
                n_elems, "number", "n_elems",
                "specify vector size in elements for both A and B. Each "
                "element is of 8B. Default uses 1M elements",
                "1048576"))},
      {'t', OptionPtr(new Range(numThreads, 1,
                                (uint64_t)std::numeric_limits<uint64_t>::max,
                                "threads", "number of CUDA threads", "1024"))},
      {'b', OptionPtr(new Range(blkSize, 1,
                                (uint64_t)std::numeric_limits<uint64_t>::max,
                                "blk_size", "CUDA thread block size", "64"))},
      {'g', OptionPtr(new Option<uint32_t>(cudaDevice, "number", "gpu",
                                           "specify CUDA device", "0"))},
      {'k', OptionPtr(new Option<uint32_t>(n_ctrls, "number", "n_ctrls",
                                           "specify number of NVMe controllers",
                                           "1"))},
      {'p', OptionPtr(new Range(pageSize, 1,
                                (uint64_t)std::numeric_limits<uint64_t>::max,
                                "page_size", "size of page in cache", "4096"))},
      {'d', OptionPtr(new Range(queueDepth, 2, 65536, "queue_depth",
                                "queue depth per queue", "16"))},
      {'q', OptionPtr(new Range(numQueues, 1, 65536, "num_queues",
                                "number of queues per controller", "1"))},
      {'M', OptionPtr(new Option<uint64_t>(
                maxPageCacheSize, "number", "maxPCSize",
                "Maximum Page Cache size in bytes", "8589934592"))},
      {'P', OptionPtr(new Option<uint64_t>(
                stride, "number", "STRIDE",
                "Hashing stride factor for cc. It is calculated as P = stride. "
                "Assumes power of 2",
                "1"))},
      {'C', OptionPtr(new Option<uint64_t>(coarse, "number", "COARSE",
                                           "Thread coarsening factor", "1"))},
  };

  string optionString;
  vector<option> options;
  createLongOptions(options, optionString, parsers);

  int index;
  int option;
  OptionMap::iterator parser;

  while ((option = getopt_long(argc, argv, optionString.c_str(), &options[0],
                               &index)) != -1) {
    switch (option) {
      case '?':
        throw string("Unknown option: `") + argv[optind - 1] + string("'");

      case ':':
        throw string("Missing argument for option `") + argv[optind - 1] +
            string("'");

      case 'h':
        throw helpString(argv[0], parsers);

      default:
        parser = parsers.find(option);
        if (parser == parsers.end()) {
          throw string("Unknown option: `") + argv[optind - 1] + string("'");
        }
        parser->second->parseArgument(argv[optind - 1], optarg);
        break;
    }
  }
  /*
  #ifdef __DIS_CLUSTER__
      if (blockDevicePath == nullptr && controllerId == 0)
      {
          throw string("No block device or NVM controller specified");
      }
      else if (blockDevicePath != nullptr && controllerId != 0)
      {
          throw string("Either block device or NVM controller must be specified,
  not both!");
      }
  #else
      if (blockDevicePath == nullptr && controllerPath == nullptr)
      {
          throw string("No block device or NVM controller specified");
      }
      else if (blockDevicePath != nullptr && controllerPath != nullptr)
      {
          throw string("Either block device or NVM controller must be specified,
  not both!");
      }
  #endif

      if (blockDevicePath != nullptr && doubleBuffered)
      {
          throw string("Double buffered reading from block device is not
  supported");
      }
  */
  verifyCudaDevice(cudaDevice);
  // verifyNumberOfThreads(numThreads);

  setBDF(*this);
}

Settings::Settings() {
  cudaDevice = 0;
  cudaDeviceId = 0;
  blockDevicePath = nullptr;
  controllerPath = nullptr;
  controllerId = 0;
  adapter = 0;
  segmentId = 0;
  nvmNamespace = 1;
  doubleBuffered = false;
  numReqs = 1;
  numPages = 1024;
  startBlock = 0;
  stats = false;
  input = nullptr;
  input_a = nullptr;
  input_b = nullptr;
  output = nullptr;
  afileoffset = 0;
  bfileoffset = 0;
  type = 1;
  memalloc = 2;
  numThreads = 1024;
  blkSize = 64;
  domain = 0;
  bus = 0;
  devfn = 0;
  n_ctrls = 1;
  queueDepth = 1024;
  numQueues = 128;
  pageSize = 4096;
  numElems = 2147483648;
  repeat = 32;
  src = 0;
  maxPageCacheSize = 8589934592;
  stride = 1;
  coarse = 1;
  n_elems = 1048576;
}

#endif
