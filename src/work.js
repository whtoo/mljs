class Person {
    constructor(name,age) {
        this.name = name
        this.age = age
    }

    sayHi() {
        console.log("Say hello with "+this.name)
    }    
}

function checkName(person) {
    if(person instanceof Person) {
        person.sayHi()
    } else {
        throw new Error("It's not a sub-class from Person")
    }
    
}


export {
    Person,
    checkName
}