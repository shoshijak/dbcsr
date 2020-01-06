project: DBCSR Library
project_github: https://github.com/cp2k/dbcsr
project_website: https://dbcsr.cp2k.org
github: https://github.com/cp2k/dbcsr
summary: ![DBCSR](|media|/logo/logo.png)
         {: style="text-align: center" }
author: DBCSR Authors
src_dir: ./src
output_dir: ./doc
fpp_extensions: F
fixed_extensions:
extensions: F
include: @CMAKE_SOURCE_DIR@/src
         @CMAKE_SOURCE_DIR@/src/base
predocmark: >
media_dir: @CMAKE_SOURCE_DIR@/docs/media
page_dir: @CMAKE_SOURCE_DIR@/docs
docmark_alt: #
predocmark_alt: <
display: public
         protected
         private
source: false
graph: false
search: true
favicon: @CMAKE_SOURCE_DIR@/docs/media/logo/logo.png