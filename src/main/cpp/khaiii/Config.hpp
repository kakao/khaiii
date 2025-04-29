/**
 * @author  Jamie (jamie.lim@kakaocorp.com)
 * @copyright  Copyright (C) 2018-, Kakao Corp. All rights reserved.
 */


#ifndef SRC_MAIN_CPP_KHAIII_CONFIG_HPP_
#define SRC_MAIN_CPP_KHAIII_CONFIG_HPP_


//////////////
// includes //
//////////////
#include <memory>
#include <string>
#include <unordered_map>

#include "nlohmann/json.hpp"


namespace khaiii {


/**
 * JSON format configuration file
 */
class Config {
 public:
    int class_num = -1;    ///< number of classes
    int embed_dim = -1;    ///< embedding dimension
    int hidden_dim = -1;    ///< hidden dimension
    int vocab_size = -1;    ///< vocabulary size
    int window = -1;    ///< context window size

    bool preanal = true;    ///< whether apply preanal or not
    bool errpatch = true;    ///< whether apply error patch or not
    bool restore = true;    ///< whether restore morphemes or not

    Config() = default;
    Config(const Config&) = delete;    ///< delete copy constructor
    Config& operator=(const Config&) = delete;    ///< delete assignment operator

    /**
     * 파일로부터 설정을 읽어들인다.
     * @param  path  file path
     */
    void read_from_file(const char* path);

    /**
     * JSON 옵션을 이용해 설정을 override 한다.
     * @param  opt_str  option string (JSON format)
     */
    void override_from_str(const char* opt_str);

    /**
     * 객체를 복사하고 설정을 override 한다.
     * @param  opt_str option string (JSON format)
     * @return  존재할 경우 그 옵션 객체
     */
    Config* copy_and_override(const char* opt_str);

    /**
     * 파싱된 JSON 객체를 이용해서 멤버를 세팅한다.
     * @param  jsn  JSON 객체
     */
    void set_members(const nlohmann::json& jsn);

    /**
     * 파싱된 JSON 객체를 이용해서 오버라이딩할 멤버만 세팅한다.
     * @param  jsn  JSON 객체
     */
    void override_members(const nlohmann::json& jsn);

    /**
     * 자기 자신을 복사한 객체를 생성한다.
     * @return  복사된 객체
     */
    std::shared_ptr<Config> copy();

 private:
    /**
     * 오버라이딩된 객체의 캐시
     */
    std::unordered_map<std::string, std::shared_ptr<Config>> _cfg_cache;
};


}    // namespace khaiii


#endif  // SRC_MAIN_CPP_KHAIII_CONFIG_HPP_
