from numpy.core.einsumfunc import einsum
import pyswip
import threading
import sys
import pathlib

#graph_data test
from gnn import GraphData, extractActions, exctractImage, input2actionGraph, input2graph, gnn_output_sign
data=[GraphData() for i in range(2)]
data[0].load_from_str("1 1 1 1 1 1 1 1 0 1 0 1,0 1 2 3 4 3 1 0 3 3,-1 -1 -1 -1 -1 -1 2 -1 -1 -1 4 -1 -1 -1 -1 -1 8 -1 10 -1,-1 1 1 -1 1 1 -1 1 1 -1;0 0 1 0 1 0 0 0 1 0 1 0,3 3 3 3,3 -1 5 -1 9 -1 11 -1,-1 1 1 -1;0 0 0 0 0 0 0 0 0 0 0 0,,,;2 2 1 4 1,0 -1 -1 7 -1 -1 1 -1 -1 6 -1 -1 2 -1 -1 3 2 -1 5 4 -1 9 8 -1 11 10 -1 4 -1 -1,-1 1 1 -1 1 -1 1 1 -1 1;2 2 0 1 0 1 2 1 0 1 0 1,0 1 2 4 2 3 3 5 4 4 5;1 1 2 2 3 2,0 0 1 3 5 6 7 1 9 6 11;0 0 1 0 1 0 0 0 3 0 3 0;1 1 0 1 0;0 1 3 3 3 3;0 0 0 0 0 0 1 0 0 0 0")
data[1].load_from_str('1 1 1 1 1 1 1 0 1 1 0 1 1 0 0 1 1 1 1 1 0 1 1 0 0 1 1 1 1 1 0 1 0 1 1 1 0 1 0 0 1 1 0 0 1 0 1 1 0 0 1 0 0 1 1 1 1 0 0 1 1 1 1 0 0 1 0 0 1 1 1 1,0 0 1 2 3 2 4 2 4 3 4 2 3 3 2 4 5 4 2 5 5 3 4 2 2 4 4 4 4 4 4 4 4 5 5 4 4 4 4 2 2 4 3 3 4 4 4,-1 -1 -1 -1 -1 -1 2 -1 3 3 3 -1 4 5 7 -1 2 8 10 2 11 10 14 -1 13 15 13 14 17 -1 16 18 20 2 21 2 24 -1 23 25 23 24 27 23 26 28 30 -1 32 -1 31 33 30 32 36 36 38 39 39 38 42 43 42 45 45 43 48 49 51 52 50 53 48 51 49 52 57 58 57 -1 58 -1 60 61 63 64 66 67 65 68 63 66 64 67,-1 1 1 1 1 1 1 1 1 1 -1 1 1 1 1 -1 1 -1 1 1 1 1 -1 1 1 1 -1 -1 1 -1 -1 1 1 1 1 -1 1 1 1 1 1 -1 1 1 -1 1 1;0 0 2 2 1 0 0 1 0 0 1 1 0 2 1 0 1 1 0 0 1 1 0 2 1 0 1 1 0 0 2 1 1 0 0 0 1 0 1 1 0 0 2 0 0 1 0 0 2 1 1 1 0 0 0 0 0 2 1 0 1 0 0 2 1 1 1 0 0 0 0 0,2 4 3 2 4 2 3 4 3 3 2 4 2 5 4 5 5 2 4 3 2 4 4 2 4 4 4 4 4 4 5 4 4 4 5 4 2 2 4 3 4 4 4 3,3 -1 9 8 4 3 5 -1 6 5 8 -1 11 2 12 10 16 15 17 14 15 -1 19 18 18 -1 21 2 22 2 26 25 27 24 25 -1 29 28 28 23 31 -1 35 32 34 33 33 -1 37 36 40 39 41 38 44 43 46 45 47 43 50 49 55 51 56 52 54 53 53 52 59 58 60 -1 61 -1 62 61 65 64 70 66 71 67 69 68 68 67,1 1 1 1 1 1 1 -1 1 1 1 -1 1 1 -1 1 1 1 -1 1 1 -1 1 1 -1 1 -1 -1 1 1 1 1 1 -1 1 1 1 1 -1 1 1 1 -1 1;0 0 3 1 0 1 0 0 1 0 1 0 0 0 1 1 0 0 1 0 0 0 0 1 1 1 0 0 1 0 0 0 1 1 0 0 1 0 1 1 0 0 0 2 0 1 0 0 0 1 0 1 2 1 0 0 0 0 1 0 0 1 0 0 1 0 1 2 1 0 0 0,3 5 4 3 4 4 4 3 3 4 3 5 5 4 4 4 4 4 4 4 4 4 5 4 5 4 4 4 4 3 4 3 4 4,11 10 21 20 22 21 4 3 6 4 9 2 12 11 17 13 16 13 19 16 28 27 27 23 26 23 29 26 35 30 34 31 37 36 41 39 40 38 44 42 47 45 46 42 50 48 55 48 53 51 56 49 54 50 59 57 62 60 65 63 70 63 68 66 71 64 69 65,1 1 -1 1 1 1 -1 1 1 -1 1 1 1 -1 -1 1 -1 -1 1 -1 1 1 1 1 1 1 -1 1 -1 1 1 1 1 -1;2 1 10 7 22 5,0 -1 -1 1 -1 -1 2 -1 -1 3 2 -1 5 3 -1 8 7 -1 15 14 -1 18 17 -1 25 24 -1 31 30 -1 33 32 -1 60 57 -1 61 58 -1 4 3 3 11 10 2 16 13 15 17 13 14 28 27 23 65 63 64 68 66 67 6 4 5 9 2 8 12 11 10 19 16 18 22 21 2 29 26 28 34 31 33 35 30 32 37 36 36 40 38 39 41 39 38 44 42 43 46 42 45 47 45 43 54 50 53 55 48 51 56 49 52 59 57 58 62 60 61 69 65 68 70 63 66 71 64 67 21 20 2 26 23 25 27 23 24 50 48 49 53 51 52,-1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 -1 -1 -1 -1 1 -1 -1 1 -1 -1 1 1 -1 1 1 1 -1 -1 1 1 1 1 1 1 1;3 1 0 0 0 0 1 0 0 1 0 0 1 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 1 0 0 0 0 1 1 0 1 0 0 1 1 0 0 1 0 1 1 0 0 0 0 0 0 1 1 1 0 0 1 0 0 1 0 0 0 0 0 0 1 1 1,0 1 3 2 2 4 5 6 7 8 9 9 10 11 11 12 12 12 13 13 13 14 14 15 15 15;1 1 2 1 1 1 1 1 1 2 1 2 3 3 2 3,0 0 1 6 0 9 12 19 22 29 34 35 37 40 41 44 46 47 54 55 56 59 62 69 70 71;0 0 1 1 1 1 0 3 1 0 3 1 0 3 3 1 1 1 1 0 3 1 0 3 3 1 1 1 1 0 3 1 3 1 0 0 3 0 3 3 0 0 3 3 0 3 0 0 3 3 1 3 3 1 0 0 0 3 3 0 1 1 0 3 3 1 3 3 1 0 0 0;1 0 0 0 1 0;0 1 3 3 3 3 3 3 3 3 3 3 3 3 3 3;0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0')
actionData=[GraphData() for i in range(1)]
actionData[0].load_from_str('1 1 1 1 1 1 1 1 1 1 1 3 1 1 1 1 1 1 2 1 1 2 2 1 1 0 1 0 1 2 2 1 1 1 1 0 0 0 1 1 1 1 0 0 0 1 1 1 1 0 0 0 1 1 1 1 0 0 1 1 1 1 1 0 0 0 2 1 1 1 1 0 0 1 1 0 2 1 0 1 1 0 0 1 1 1 1 1 0 0 1 1 0 2 1 1 0 0 1 0 1 1 1 0 1 0 1 0 1 1 1 1 0 3 1 0 1 4 1 0 1 0 1 0 1 1 1 1 0 3 1 0 1 4 1 2 1 0 1 0 1 0 1 1 1 1 0 3 1 0 1 4 2 1 0 1 0 1 0 1 1 1 1 0 3 1 3 1 0 1 0 1 0 1 1 1 1 0 3 1 0 1 1 2 1 2 3 1 0 0 1 0 0 1 1 1 1 0 0 1 1 1 1 0 0 0 0 3 0 0 0 0 3 1 1 1 1 1 0 0 1 0 0 1 1 1 1 0 0 1 0 0 1 1 1 1 0 0 0 2 0 0 0 2 1 1 1 1 0 0 1 1 1 1 0 0 0 2 0 0 0 2 1 1 1 1 0 1 0 0 1 1 0 0 1 0 1 1 0 0 1 0 0 1 1 1 0 0 1 0 0 1 1 1 0 1 0 1 1 0 1 0 1 1 0 1 0 1 1 0 1 0 1 1 0 0 1 0 0 1 1 1 0 0 1 0 0 1 1 1,0 1 2 3 2 4 5 6 7 8 9 10 10 10 11 12 8 13 8 14 15 15 16 16 17 17 17 17 18 11 8 8 17 17 17 17 18 16 16 0 6 7 8 19 6 7 8 20 6 7 8 21 14 7 8 19 20 17 17 8 19 21 5 7 8 15 15 8 22 8 14 23 18 19 20 7 8 15 15 24 18 8 16 7 8 2 2 6 7 8 5 10 10 10 11 8 25 25 25 25 8 2 2 6 7 8 5 10 10 10 11 8 25 25 25 25 14 15 15 16 2 2 6 7 8 5 10 10 10 11 8 25 25 25 25 17 17 18 2 2 6 7 8 5 10 10 10 11 26 26 26 8 2 2 6 7 8 5 10 10 10 11 8 14 15 15 16 17 17 26 26 26 18 14 14 18 18 18 18 7 7 18 10 10 10 10 10 10 18 18 18 18 18 6 6 18 18 18 24 24 18 18 18 17 17 17 17 18 18 18 18 18 23 23 18 15 15 15 15 18 18 18 18 18 18 18 18 18 18 8 8 18 18 16 16 18 18 5 18 5 19 18 19 2 18 2 11 18 11 20 20 18 18 21 21 18 18,-1 -1 -1 -1 1 -1 -1 -1 3 -1 -1 -1 5 -1 3 1 7 -1 5 8 -1 -1 3 1 1 5 5 10 11 -1 -1 -1 13 3 -1 -1 15 3 3 5 3 10 10 17 13 18 15 18 1 5 5 13 1 5 5 15 21 22 11 -1 25 3 27 3 1 5 5 25 1 5 5 27 29 30 25 18 27 18 -1 -1 36 37 38 -1 35 39 35 -1 43 44 45 -1 42 46 42 43 50 51 52 -1 49 53 49 51 56 57 56 -1 58 59 57 -1 57 56 63 64 64 65 66 63 64 -1 64 63 64 -1 72 -1 71 73 72 75 75 71 76 73 78 -1 79 78 81 82 82 -1 83 84 82 -1 82 81 89 -1 88 90 89 92 92 88 92 88 93 94 96 97 96 99 97 -1 99 101 103 -1 105 -1 105 103 108 -1 107 109 107 -1 105 103 103 107 107 112 113 -1 115 103 115 107 107 105 105 103 103 112 117 105 119 -1 121 -1 121 119 124 -1 123 125 123 -1 121 119 119 123 123 128 129 -1 131 119 131 123 123 121 121 119 119 128 121 123 121 128 128 134 133 135 137 -1 139 -1 139 137 142 -1 141 143 141 -1 139 137 137 141 141 146 147 -1 149 137 149 141 141 139 139 137 137 146 137 141 141 151 152 149 154 -1 156 -1 156 154 159 -1 158 160 158 -1 156 154 154 158 158 163 164 -1 158 156 156 154 154 163 166 154 168 -1 170 -1 170 168 173 -1 172 174 172 -1 170 168 168 172 172 177 178 -1 180 170 170 172 170 177 177 182 180 183 168 172 172 180 172 170 170 168 168 177 185 186 188 189 191 192 190 193 188 191 189 192 197 198 197 -1 198 -1 200 201 203 204 204 205 205 206 208 209 209 210 210 211 207 212 203 208 204 209 205 210 206 211 218 219 221 222 220 223 218 221 219 222 227 228 230 231 229 232 227 230 228 231 236 237 237 238 240 241 241 242 239 243 236 240 237 241 238 242 248 249 248 -1 249 -1 251 252 254 255 255 256 258 259 259 260 257 261 254 258 255 259 256 260 266 266 268 269 269 268 272 273 272 275 275 273 278 279 281 282 281 278 282 279 286 287 289 290 289 286 290 287 294 -1 296 294 296 -1 299 -1 301 299 301 -1 304 -1 306 304 306 -1 309 -1 311 309 311 -1 314 315 317 318 317 314 318 315 322 323 325 326 325 322 326 323,1 1 1 1 1 1 -1 1 1 -1 1 1 1 1 1 1 -1 1 -1 1 1 1 -1 -1 1 1 1 1 1 -1 1 1 1 1 1 1 -1 1 1 -1 1 1 1 -1 1 1 1 -1 1 1 1 -1 1 1 -1 1 1 1 1 -1 1 1 1 1 1 1 1 -1 1 -1 1 1 -1 1 1 1 1 1 1 1 -1 -1 1 1 1 -1 -1 1 1 1 1 1 1 1 -1 1 1 1 1 1 -1 -1 -1 1 1 1 1 1 1 1 -1 1 1 1 1 1 1 1 1 -1 -1 -1 1 1 1 1 1 1 1 -1 1 1 1 1 1 1 1 1 -1 -1 1 1 1 1 1 1 1 1 1 1 1 -1 -1 -1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 -1 1 1 -1 1 1 1 1 1 -1 1 1 1 1 1 1 -1 1 1 1 1 1 1 -1 1 1 1 1 -1 1 1 1 1 1 1 -1 1 1 1 1 1 1 -1 1 1 1 1 -1 1 1 1 -1 1 -1 -1 1 1 -1 1 1 1 -1 1 1 1 -1 1 1 -1 1 1 -1 1 1 -1 1 1 -1 1 1 1 -1 1 1 1;0 6 0 5 0 7 0 1 0 0 1 2 0 2 0 2 0 0 0 0 0 1 0 0 0 2 0 2 0 1 0 0 0 0 0 2 1 0 1 0 0 0 2 1 0 1 0 0 0 2 1 0 1 0 0 0 2 2 1 0 0 0 0 1 4 0 1 0 0 0 0 1 2 0 0 1 1 0 1 1 0 1 3 1 0 0 0 0 1 2 0 0 2 1 0 0 2 1 0 1 0 0 0 3 0 4 0 4 1 0 0 0 0 1 0 2 0 1 0 3 0 6 0 4 1 0 0 0 1 1 0 2 0 1 0 0 0 4 0 4 0 5 1 0 0 0 0 1 0 2 0 0 1 0 3 0 4 0 4 1 0 0 0 0 1 0 1 0 4 0 6 0 5 1 0 0 0 1 1 0 2 0 0 0 0 1 0 0 2 1 1 1 0 0 0 0 0 2 1 0 1 0 0 2 2 2 1 1 1 1 1 0 0 0 0 0 0 0 2 1 1 1 0 0 0 0 0 2 1 1 1 0 0 0 0 0 2 2 1 1 1 1 0 0 0 0 0 0 2 1 0 1 0 0 2 2 1 1 1 1 0 0 0 0 0 0 1 0 1 1 0 0 2 0 0 1 0 0 1 0 0 2 1 0 0 0 1 0 0 2 1 0 0 0 1 0 2 0 0 1 0 2 0 0 1 0 2 0 0 1 0 2 0 0 1 0 0 2 1 0 0 0 1 0 0 2 1 0 0 0,2 10 17 17 17 17 2 6 10 14 15 5 8 10 17 17 17 17 7 15 11 11 8 16 8 16 18 8 16 8 16 18 8 19 6 7 8 20 6 7 8 21 6 7 14 7 19 20 8 17 17 19 21 5 8 8 7 15 15 8 22 8 14 23 19 20 18 8 7 15 15 24 18 8 16 7 8 2 10 25 2 6 10 25 8 5 10 25 7 11 8 25 8 2 10 25 2 6 10 25 14 15 8 5 10 25 7 15 11 8 25 16 2 10 25 17 2 6 10 25 8 5 10 25 17 7 11 8 25 18 2 10 26 2 6 10 26 8 5 10 26 7 11 8 2 10 17 26 2 6 10 14 15 26 8 5 10 17 26 7 15 11 8 16 18 14 18 18 18 14 18 7 7 18 10 18 10 18 10 18 18 18 10 10 10 6 18 18 18 6 24 18 18 18 24 17 18 17 18 18 18 17 17 18 23 23 18 15 18 15 18 18 18 15 15 18 18 18 18 18 18 8 8 18 18 16 16 18 18 5 18 5 19 18 19 2 18 2 11 18 11 20 20 18 18 21 21 18 18,2 -1 11 5 21 5 22 5 29 5 30 5 4 -1 7 1 11 1 17 5 18 10 6 -1 9 8 11 10 21 13 22 15 29 25 30 27 8 -1 18 17 12 -1 24 -1 14 3 19 18 16 3 20 18 23 22 26 3 32 18 28 3 33 18 31 30 40 39 41 -1 38 37 39 -1 47 46 48 43 45 44 46 -1 54 53 55 51 52 51 53 -1 58 57 59 -1 61 -1 62 56 60 59 66 64 66 65 68 -1 69 63 70 -1 67 63 74 73 73 -1 76 75 76 71 77 73 79 -1 80 78 83 82 84 -1 86 -1 87 81 85 84 91 90 90 -1 93 92 93 88 94 88 95 94 98 97 100 99 101 -1 102 101 104 -1 113 107 117 112 106 -1 108 103 113 103 117 103 110 109 111 -1 113 112 117 105 109 -1 114 -1 116 103 117 107 118 105 120 -1 129 123 133 128 122 -1 124 119 129 119 133 119 134 123 135 128 126 125 127 -1 129 128 133 121 125 -1 135 134 130 -1 132 119 133 123 136 135 138 -1 147 141 151 146 152 141 140 -1 142 137 147 137 151 137 144 143 145 -1 147 146 151 139 152 151 143 -1 148 -1 150 137 151 141 153 149 155 -1 164 158 166 163 157 -1 159 154 164 154 166 154 161 160 162 -1 164 163 166 156 160 -1 165 -1 167 154 169 -1 178 172 185 172 186 177 171 -1 173 168 178 168 182 172 183 177 186 168 175 174 176 -1 178 177 185 180 186 170 174 -1 183 182 179 -1 181 170 184 183 187 186 190 189 195 191 196 192 194 193 193 192 199 198 200 -1 201 -1 202 201 207 204 214 208 207 205 215 209 207 206 216 210 217 211 213 212 212 209 212 210 212 211 220 219 225 221 226 222 224 223 223 222 229 228 234 230 235 231 233 232 232 231 239 237 245 240 239 238 246 241 247 242 244 243 243 241 243 242 250 249 251 -1 252 -1 253 252 257 255 263 258 257 256 264 259 265 260 262 261 261 259 261 260 267 266 270 269 271 268 274 273 276 275 277 273 280 279 283 282 284 278 285 279 288 287 291 290 292 286 293 287 295 -1 297 294 298 -1 300 -1 302 299 303 -1 305 -1 307 304 308 -1 310 -1 312 309 313 -1 316 315 319 318 320 314 321 315 324 323 327 326 328 322 329 323,1 1 1 1 1 1 1 1 1 1 1 -1 -1 1 1 1 1 1 1 1 1 -1 -1 -1 -1 -1 1 1 1 1 1 -1 1 -1 1 1 1 -1 1 1 1 -1 1 1 1 1 1 1 -1 1 1 1 1 1 -1 1 1 1 1 -1 1 -1 1 1 1 1 -1 1 1 1 1 1 -1 -1 1 1 1 -1 1 1 -1 1 1 1 1 1 1 1 1 -1 1 1 -1 -1 1 1 -1 1 1 1 1 1 1 1 1 1 1 1 -1 1 1 -1 -1 1 1 1 -1 1 1 1 1 1 1 1 1 1 -1 1 1 1 -1 1 1 -1 1 1 1 1 1 1 1 1 1 -1 -1 1 1 1 -1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 -1 1 1 1 -1 1 1 1 1 -1 1 1 1 1 1 1 1 -1 1 1 1 1 1 1 -1 1 1 1 1 -1 1 1 1 1 1 1 -1 1 1 1 1 1 -1 1 1 1 1 1 -1 1 1 -1 1 -1 -1 1 1 -1 1 1 1 -1 1 1 1 -1 1 1 -1 1 1 -1 1 1 -1 1 1 -1 1 1 1 -1 1 1 1;0 2 0 4 0 6 0 0 1 0 2 0 0 1 0 1 0 1 4 0 0 0 1 0 0 1 0 1 0 0 1 0 0 0 0 0 0 1 0 1 0 0 0 1 1 0 1 0 0 0 0 2 0 1 0 0 1 1 0 1 0 0 0 2 1 1 0 0 0 0 0 1 0 2 0 1 0 0 1 0 0 1 1 0 1 0 0 0 2 0 1 0 1 0 1 0 0 1 0 1 0 1 0 4 0 2 0 2 0 1 0 0 2 0 0 0 0 0 0 4 0 1 0 3 0 1 0 0 3 0 0 0 0 0 1 1 0 4 0 1 0 3 0 1 0 0 2 0 0 1 0 1 0 0 4 0 1 0 1 0 1 0 0 2 0 0 0 0 3 0 2 0 3 0 1 0 0 3 0 0 1 0 1 1 0 0 1 0 0 1 0 1 2 1 0 0 0 0 1 0 0 1 0 0 1 1 1 0 1 2 2 2 1 0 0 0 0 0 0 1 0 1 2 1 0 0 0 0 1 0 1 2 1 0 0 0 0 1 1 0 1 2 2 1 0 0 0 0 0 1 0 0 1 0 0 1 1 0 1 2 2 1 0 0 0 0 1 0 1 1 0 0 0 2 0 1 0 0 1 2 0 0 1 0 0 0 1 2 0 0 1 0 0 0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 2 0 0 1 0 0 0 1 2 0 0 1 0 0 0,6 10 8 8 8 8 10 14 17 17 17 17 8 10 15 17 17 15 16 16 16 16 18 17 17 18 6 8 20 6 8 6 21 8 20 14 8 8 21 17 17 15 8 8 15 8 20 14 18 15 24 8 15 18 8 16 8 6 10 8 25 25 8 10 25 8 10 25 6 10 8 25 25 10 25 14 8 10 25 15 15 16 6 10 8 25 25 10 25 17 8 10 25 18 17 6 10 26 8 26 10 8 10 26 6 10 26 8 26 10 14 17 8 10 15 26 17 15 16 18 14 18 14 18 18 18 18 10 10 10 18 10 18 10 18 10 18 18 6 18 6 18 18 24 18 24 18 18 17 17 18 17 18 17 18 18 18 18 15 15 18 15 18 15 18 18 18 18 18 18 18 18 18 8 18 8 18 16 18 16 18 18 18 18 18 20 18 20 18 21 18 21,7 3 11 3 14 13 16 15 26 25 28 27 11 1 17 3 21 1 22 1 29 1 30 1 9 5 11 5 18 3 21 5 22 5 18 10 19 13 20 15 32 25 33 27 23 21 29 5 30 5 31 29 38 36 40 35 48 42 45 43 47 42 52 50 55 49 54 49 62 57 58 56 60 58 67 66 69 64 66 63 66 64 76 75 74 71 77 76 76 72 80 79 87 82 83 81 85 83 93 92 94 92 91 88 93 89 95 93 98 96 100 96 102 99 108 105 113 105 116 115 117 105 117 107 118 117 113 103 117 115 110 107 113 107 117 103 124 121 129 121 132 131 133 121 133 123 129 119 133 131 134 121 126 123 129 123 133 119 135 121 135 128 136 133 142 139 147 139 150 149 151 139 151 141 147 137 151 149 152 137 144 141 147 141 151 137 153 152 152 141 159 156 164 156 166 156 167 166 166 158 164 154 161 158 164 158 166 154 173 170 178 170 186 170 181 180 186 172 178 168 182 170 185 168 175 172 178 172 183 170 186 168 185 172 183 177 184 180 187 185 190 188 195 188 193 191 196 189 194 190 199 197 202 200 207 203 207 204 207 205 214 203 212 208 215 204 212 209 216 205 212 210 217 206 213 207 220 218 225 218 223 221 226 219 224 220 229 227 234 227 232 230 235 228 233 229 239 236 239 237 245 236 243 240 246 237 243 241 247 238 244 239 250 248 253 251 257 254 257 255 263 254 261 258 264 255 261 259 265 256 262 257 267 266 271 269 270 268 274 272 277 275 276 272 284 281 280 278 285 282 283 281 292 289 288 286 293 290 291 289 297 296 302 301 307 306 312 311 320 317 316 314 321 318 319 317 328 325 324 322 329 326 327 325,1 1 -1 -1 1 1 1 1 1 1 1 1 -1 1 1 1 1 1 -1 -1 1 1 1 1 1 -1 1 1 -1 1 1 1 -1 1 1 1 -1 -1 1 1 1 1 1 -1 1 -1 1 1 -1 1 1 1 1 -1 -1 1 1 1 1 1 1 1 -1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 -1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 -1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 -1 1 1 1 1 -1 1 -1 1 1 1 1 1 1 1 1 1 1 -1 1 1 1 1 -1 1 1 1 1 -1 1 1 1 1 1 1 1 -1 1 -1 1 1 1 1 1 1 1 -1 -1 -1 1 -1 1 1 1 -1 1 1 1 -1 1 1 1 1 1 1 1 -1 1 1 1 -1 1 1;2 1 14 1 1 9 11 15 29 1 24 9 1 1 7 14 9 18 50 6 5 4 1 3 3 12 6,0 -1 -1 34 -1 -1 1 -1 -1 2 1 -1 4 3 -1 104 103 -1 106 105 -1 120 119 -1 122 121 -1 138 137 -1 140 139 -1 155 154 -1 157 156 -1 169 168 -1 171 170 -1 305 304 -1 308 306 -1 3 -1 -1 5 -1 -1 6 5 -1 70 64 -1 111 107 -1 127 123 -1 145 141 -1 162 158 -1 176 172 -1 295 294 -1 298 296 -1 7 3 1 38 36 37 45 43 44 52 50 51 108 105 103 124 121 119 142 139 137 159 156 154 173 170 168 220 218 219 223 221 222 8 7 -1 39 38 -1 46 45 -1 53 52 -1 59 56 -1 73 72 -1 90 89 -1 101 97 -1 109 108 -1 125 124 -1 143 142 -1 160 159 -1 174 173 -1 200 197 -1 201 198 -1 9 5 8 14 13 3 16 15 3 26 25 3 28 27 3 40 35 39 47 42 46 54 49 53 60 58 59 67 66 63 74 71 73 77 76 73 80 79 78 91 88 90 98 96 97 102 99 101 110 107 109 116 115 103 118 117 105 126 123 125 132 131 119 144 141 143 150 149 137 161 158 160 167 166 154 175 172 174 181 180 170 280 278 279 283 281 282 10 -1 -1 11 3 1 11 1 5 11 5 10 113 105 103 113 103 107 113 107 112 129 121 119 129 119 123 129 123 128 147 139 137 147 137 141 147 141 146 164 156 154 164 154 158 164 158 163 178 170 168 178 168 172 178 172 177 207 203 204 207 204 205 207 205 206 212 208 209 212 209 210 212 210 211 12 11 -1 24 11 -1 114 113 -1 130 129 -1 148 147 -1 165 164 -1 179 178 -1 310 309 -1 313 311 -1 13 -1 -1 15 -1 -1 17 3 5 58 56 57 83 81 82 134 121 123 182 170 172 190 188 189 193 191 192 18 3 10 18 10 17 76 72 75 76 75 71 93 89 92 93 92 88 135 121 128 135 128 134 183 170 177 183 177 182 257 254 255 257 255 256 261 258 259 261 259 260 19 13 18 20 15 18 32 25 18 33 27 18 100 96 99 136 133 135 184 180 183 288 286 287 291 289 290 21 1 5 21 5 13 22 1 5 22 5 15 29 1 5 29 5 25 30 1 5 30 5 27 66 63 64 66 64 65 152 137 141 152 141 151 185 168 172 185 172 180 239 236 237 239 237 238 243 240 241 243 241 242 23 21 22 31 29 30 85 83 84 95 93 94 153 152 149 187 185 186 194 190 193 195 188 191 196 189 192 199 197 198 202 200 201 213 207 212 214 203 208 215 204 209 216 205 210 217 206 211 224 220 223 225 218 221 226 219 222 233 229 232 234 227 230 235 228 231 244 239 243 245 236 240 246 237 241 247 238 242 250 248 249 253 251 252 262 257 261 263 254 258 264 255 259 265 256 260 267 266 266 270 268 269 271 269 268 274 272 273 276 272 275 277 275 273 284 281 278 285 282 279 292 289 286 293 290 287 297 296 294 302 301 299 307 306 304 312 311 309 320 317 314 321 318 315 328 325 322 329 326 323 41 35 -1 61 57 -1 68 64 -1 86 82 -1 300 299 -1 303 301 -1 48 42 43 62 57 56 87 82 81 316 314 315 319 317 318 55 49 51 69 64 63 324 322 323 327 325 326 79 78 -1 84 82 -1 251 248 -1 252 249 -1 94 92 88 229 227 228 232 230 231 117 115 107 117 107 105 117 105 103 117 103 112 133 131 123 133 123 121 133 121 119 133 119 128 151 149 141 151 141 139 151 139 137 151 137 146 166 158 156 166 156 154 166 154 163 186 172 170 186 170 168 186 168 177,1 -1 1 1 1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 1 1 1 -1 1 1 1 1 1 1 -1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 -1 -1 -1 1 1 1 1 1 -1 -1 1 -1 -1 1 -1 1 1 1 -1 1 1 1 1 1 -1 1 1 -1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 -1 -1 -1 -1 1 1 -1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 -1 -1 1 1 1 -1 1 -1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 -1 -1 -1 1 -1 -1 1 1 1 -1 -1 1 1 1 1 -1 1 1 -1 1 1 -1 1 1 1 1 -1 -1 1 1 1 -1 1 -1 -1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 -1 1 1 1 -1 1 -1 1 1 -1 1 -1 1 -1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1;10 0 1 0 1 0 1 0 0 1 0 0 5 0 1 0 1 0 0 1 1 0 0 1 1 0 1 0 1 0 0 1 1 1 1 0 0 0 0 0 1 1 0 0 0 0 0 1 1 0 0 0 0 0 1 1 0 0 0 0 1 1 1 0 0 0 0 1 1 1 1 0 0 0 1 0 0 1 0 0 1 0 0 0 0 1 1 1 0 0 0 1 0 0 0 1 0 0 1 0 1 0 1 0 1 0 1 0 0 0 1 1 0 0 1 0 1 0 1 0 1 0 1 0 0 0 1 1 0 0 1 0 1 0 0 0 1 0 1 0 1 0 0 0 1 1 0 0 1 0 1 0 0 1 0 1 0 1 0 0 0 1 1 0 0 1 0 1 0 1 0 1 0 0 0 1 1 0 0 1 0 1 0 0 1 0 0 1 0 0 0 0 0 0 1 1 1 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 1 1 1 1 0 0 1 0 0 1 0 0 0 0 0 0 0 0 1 1 1 1 0 1 0 0 1 1 0 0 1 0 1 1 0 0 1 0 0 1 1 1 0 0 1 0 0 1 1 1 0 1 0 1 1 0 1 0 1 1 0 1 0 1 1 0 1 0 1 1 0 0 1 0 0 1 1 1 0 0 1 0 0 1 1 1,0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 4 5 6 7 8 9 9 9 9 9 9 10 11 11 12 12 13 13 14 14 14 15 15 15 15 16 16 17 18 18 18 19 19 20 20 20 21 21 21 21 21 21 21 22 22 22 22 22 22 22 23 23 23 23 23 23 23 24 24 24 24 24 24 25 25 25 25 25 25 25 25 26 26 26 27 27 28 28 28 28 28 29 29 29 30 30 30 31 31 31 31 32 32 33 33 33 33 34 35 35 36 36 36 37 37 37 37 38 38 38 38 39 39 39 40 40 40 41 41 41 42 42 42 43 43 43 43 44 44 44 44;2 2 2 2 3 3 3 3 3 7 1 2 2 2 3 4 2 1 3 2 3 7 7 7 6 8 3 2 5 3 3 4 2 4 1 2 3 4 4 3 3 3 3 4 4,0 2 0 4 0 6 0 9 0 12 14 0 12 16 0 12 19 0 12 20 0 12 23 0 24 26 28 31 32 33 34 40 41 47 48 54 55 60 61 62 67 68 69 70 74 77 80 85 86 87 91 95 98 100 102 104 106 110 111 114 116 118 120 122 126 127 130 132 136 138 140 144 145 148 150 153 155 157 161 162 165 167 169 171 175 176 179 181 184 187 194 195 196 199 202 213 214 215 216 217 224 225 226 233 234 235 244 245 246 247 250 253 262 263 264 265 267 270 271 274 276 277 280 283 284 285 288 291 292 293 295 297 298 300 302 303 305 307 308 310 312 313 316 319 320 321 324 327 328 329;0 1 0 1 0 1 0 1 1 0 1 1 0 1 0 1 0 1 1 0 0 1 1 0 0 3 0 3 0 1 1 0 0 0 0 3 3 3 1 1 0 0 3 3 3 1 1 0 0 3 3 3 1 1 0 0 3 3 1 1 0 0 0 3 3 3 1 0 0 0 0 3 3 1 0 3 1 0 3 1 0 3 3 1 1 0 0 0 3 3 1 0 3 1 1 0 3 3 0 3 0 1 0 3 0 3 0 3 1 1 0 0 3 1 0 3 0 1 0 3 0 3 0 3 1 1 0 0 3 1 0 3 0 1 1 1 0 3 0 3 0 3 1 1 0 0 3 1 0 3 0 1 1 0 3 0 3 0 3 1 1 0 0 3 1 0 1 0 3 0 3 0 3 1 1 0 0 3 1 0 3 0 1 1 0 1 1 0 3 3 1 3 3 1 0 0 0 3 3 0 1 1 0 3 3 3 3 1 3 3 3 3 1 0 0 0 0 0 3 3 1 3 3 1 0 0 0 3 3 1 3 3 1 0 0 0 3 3 3 1 3 3 3 1 0 0 0 0 3 3 0 1 1 0 3 3 3 1 3 3 3 1 0 0 0 0 3 0 3 3 0 0 3 3 0 3 0 0 3 3 0 3 3 0 0 0 3 3 0 3 3 0 0 0 3 0 3 0 0 3 0 3 0 0 3 0 3 0 0 3 0 3 0 0 3 3 0 3 3 0 0 0 3 3 0 3 3 0 0 0;1 0 1 0 0 1 0 0 1 0 0 1 0 0 0 0 1 0 1 1 1 1 0 0 0 0 0;3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3;1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1')
#import gym
import numpy as np
import tensorflow as tf

class ProblemLibrary:
    def __init__(self, config=None):
        #self.problem=lambda: "leancop/robinson_1p1__2.p"
        #self.problem=lambda: "leancop/pelletier21.p"
        directory="leancop/theorems/m2n140"
        self._load(directory)
        print(f'Found {self.total} problem files.')
        self.problem=lambda: "/".join(str(self.problems[np.random.randint(self.total)]).split("\\"))



    def _load(self, directory):
        directory = pathlib.Path(directory).expanduser()
        self.problems=[]
        for filename in reversed(sorted(directory.glob('*.p'))):
            self.problems.append(filename)
        self.total=len(self.problems)

    def get(self):
        return self.problem()

class ProLog:
    LOCK=threading.Lock()
    
    def __init__(self, gnn=True):
        #problems is a generator function
        self.gnn=gnn

        self.step_limit=25
        self.steps=0
        
        self.problems=ProblemLibrary()

        self.step_reward = 0.2
        self.success_reward = 1
        self.failure_reward = -0.2
        self.invalid_reward = -1
        self.step_limit_reward = -0.5
        
        #with self.LOCK:
        self.prolog = pyswip.Prolog()
        self.prolog.consult("leancop/leancop_step.pl")
        # self.settings = "[conj, nodef, verbose, print_proof]"
        self.settings = "[conj, nodef]"
        problem=self.problems.get()
        query = 'init_python("{}",{},GnnInput, SimpleFeatures, Result)'.format(problem, self.settings)
        #print("Query:\n   ", query, "\n")
        result = list(self.prolog.query(query))[0]
        self.result = result["Result"]
        self.gnnInput = result["GnnInput"]
        self.simple_features = result["SimpleFeatures"]

        self.ext_action_size = len(self.gnnInput[4])
        #TODO action_perm is redundant
        self.action_perm = self.gnnInput[5]

    @property
    def action_space_size(self)->int:
        return self.ext_action_size
    
    def step(self,action):
        self.steps+=1

        #TODO output obs,reward,done, info
        #TODO BUG#001
        '''if(action.shape!=(1)):
            action=np.argmax(action)'''

        #print(action)
        if(self.gnnInput[4][action]==0):
            action=-1
        elif(action==0):
            action=0
        else:
            action=np.array(self.gnnInput[4][:action]).sum()
    
        
        #action=0
        query = 'step_python({}, GnnInput, SimpleFeatures, Result)'.format(action)
        #print("Query:\n   ", query, "\n")
        result = list(self.prolog.query(query))
        #print(result)
        if len(result) == 0:
            self.result=-1
            reward = self.invalid_reward
        else:
            self.result = result[0]["Result"]
            self.gnnInput = result[0]["GnnInput"]
            self.simple_features = result[0]["SimpleFeatures"]
            self.action_perm = self.gnnInput[5]
            reward=self.reward()
            
        obs = self.image(True)
        #print('Observation meta:', len(obs['axiom_mask']), obs['action_space']['num_clauses'], obs['action_space']['num_nodes'])
        return (obs, reward, self.terminal(), {}) 
    
    def reset(self):
        self.steps=0
        
        problem=self.problems.get()
        #remove out '.p' and /
        self.current_problem="".join(problem[:-2].split('/')[2:])
        print('Loaded problem:', self.current_problem)
        query = 'init_python("{}",{},GnnInput, SimpleFeatures, Result)'.format(problem, self.settings)
        #print("Query:\n   ", query, "\n")
        result = list(self.prolog.query(query))[0]
        self.result = result["Result"]
        self.gnnInput = result["GnnInput"]
        #result[0] -> result
        self.simple_features = result["SimpleFeatures"]

        self.ext_action_size = len(self.gnnInput[4])
        self.action_perm = self.gnnInput[5]

        obs = self.image(True)
        #print('Observation meta:', len(obs['axiom_mask']), obs['action_space']['num_clauses'], obs['action_space']['num_nodes'])
        return obs

    def image(self, reset=False):

        image={'image':np.tanh(np.array(self.simple_features,np.float32)*0.1),
            'axiom_mask':self.gnnInput[4]}
        debug=0
        if self.gnn:
            if debug==0:
                image['gnn']=exctractImage(self.prolog, self.gnnInput)
            elif debug==2:
                image['gnn']=data[0].convert_to_dict()
            elif debug==1:
                image['gnn']=data[np.random.randint(2)].convert_to_dict()
            
        if self.gnn:
            if debug==0:
                action_space=extractActions(self.prolog, self.gnnInput)
            elif debug==2:
                action_space=actionData[0].convert_to_dict()
            elif debug==3: 
                action_space=np.zeros((4,4))

        #if self.steps: image['action_space']=None
        #else: image['action_space']=action_space
        image['action_space']=action_space

        #return {'image':self.gnnInput, 'ram': None, 'features': self.get_features()},
        #return {'image':np.zeros(16)}
        return image

    def terminal(self)->bool:
        return self.result != 0

    def reward(self):
        if self.result == -1:
            reward = self.failure_reward
        elif self.result == 1:
            reward = self.success_reward
        else:
            if(self.steps==self.step_limit):
                self.result=-1
                reward = self.step_limit_reward
            else:
                reward = self.step_reward
        return np.float64(reward)

    def output_sign(self, batch_size=0, batch_length=None):
        _shape=(batch_size, batch_length) if batch_size!=0 else (batch_length,)

        spec=lambda x ,dt: tf.TensorSpec(shape=_shape+x, dtype=dt)
        sign={
            'image': spec((14,), tf.float32),
            #'features': spec((14,), tf.float32),
            'axiom_mask': spec((None,), tf.int32)
            }

        include_num=True
        """if self.gnn:
            #spec=lambda x: tf.RaggedTensorSpec(shape=(None, None,)+x, dtype=tf.int32)
            sign['gnn']=self.gnn_output_sign(lambda x: tf.TensorSpec(shape=_shape+x, dtype=tf.int32), include_num)
            sign['action_space']=self.gnn_output_sign(lambda x: tf.TensorSpec(shape=_shape+x, dtype=tf.int32), include_num)"""

        if self.gnn:
            sign['gnn']=gnn_output_sign(lambda x: tf.RaggedTensorSpec(shape=_shape+x, dtype=tf.int32), include_num)
            sign['action_space']=gnn_output_sign(lambda x: tf.TensorSpec(shape=()+x, dtype=tf.int32), include_num)

        return sign