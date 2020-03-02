# Decorator

## Examples
### 1. Testable

```
function Testable(target) {
    target.isTestable = true
}
@Testable
function add(x,y) {...}

console.log(add) /// true
```