import { Person } from "./work"

const logWrapper = targetClass =>{
    const orignRender = targetClass.prototype.render;
    targetClass.prototype.render = function(){
        console.log("wrap log begin")
        orignRender.apply(this);//防止this指向改变了
        console.log("wrap log end")
    }
    return targetClass;
}

@logWrapper
class App {
    get state(){
        return 666
    }
    render(){
        console.log("this is App's render func,state is "+ this.state);
    }
}

function testable(target) {
    target.isTestable = true
}

class Worker extends Person {
    constructor(name,age,salary){
        super(name,age)
        this.salary = salary
    }

    payWeek() {
        return this.salary * 5
    }
}

let arx = new Worker("Arx",18,2000)
arx.sayHi()
console.log(arx.payWeek())
console.log(arx.isTestable == true)

console.log(Person.isTestable == true)

let app = new App()
app.render()