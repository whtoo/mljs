# export and import

## 1. export
```
/// Assumed the codes locate in work.js
let n = function() {
    ....
}
export {n as m} ✅
// But
export n ❌
```

## 2. import
```
import { m } from './work' ✅
import { m } from 'work' ❌
                    ^
                    |
        默认的查找目录是node_modules

```
